from capturar_juego import GeometryDashCapture
from gd_lector_estado import GDStateReader
from collections import deque
import time
from typing import Optional
import numpy as np
import keyboard


fps = 60
frame_buffer = 4
reward_1 = 0.1  # por sobrevivir
checkpoint_reward = 1  # por pasar múltiplos de step_checkpoint
step_checkpoint = 3
reward_100 = 100  # por pasar el nivel
penalty_death = -10  # por morir
hold_penalty = -0.01  # por mantener espacio
reset_timeout = 5.0
reset_stable_time = 0.3
reset_max_time = 0.5


class GD_env:
    VEHICLE_CUBE = 0
    VEHICLE_SHIP = 1

    def __init__(self):
        self.capture = GeometryDashCapture()
        self.reader = GDStateReader()
        self.buffer = deque(maxlen=frame_buffer)
        self.holding = False
        self.last_step_time = 0.0
        self.step_dt = 1 / fps
        self.last_percent = None
        self.last_percent_time = 0.0
        self.percent_frozen = False
        self.freeze_threshold = 0.08  # 80 ms
        self.prev_state = None
        self.max_percent = 0.0
        self.current_vehicle = None
        self.hold_penalty_cube = hold_penalty
        self.hold_penalty_ship = 0.0

    def close(self):
        if self.holding:
            keyboard.release('space')
            self.holding = False
        self.capture.close()
        self.reader.close()

    def send_input(self, action: int):
        if action not in (0, 1):
            return
        if action == 1 and not self.holding:
            keyboard.press('space')
            self.holding = True
        elif action == 0 and self.holding:
            keyboard.release('space')
            self.holding = False

    def _get_percent(self, state):
        return float(state.get("percent", 0.0))

    def reward_function(self, prev: Optional[dict], act: dict):
        prev = prev or act
        reward = reward_1

        prev_checkpoint = int(self._get_percent(prev) / step_checkpoint)
        act_checkpoint = int(self._get_percent(act) / step_checkpoint)
        if act_checkpoint > prev_checkpoint:
            reward = checkpoint_reward

        if act.get("attempts", 0) > prev.get("attempts", 0):
            reward = penalty_death

        if act.get("percent", 0) > 99:
            reward += reward_100

        if self.holding:
            vehicle = prev.get("vehicle", act.get("vehicle", self.VEHICLE_CUBE))
            if vehicle == self.VEHICLE_SHIP:
                reward += self.hold_penalty_ship
            else:
                reward += self.hold_penalty_cube

        return reward

    def is_done(self, prev: Optional[dict], act: dict):
        prev = prev or act
        if act.get("vehicle") == self.VEHICLE_SHIP and self.percent_frozen:
            return True
        if act.get("attempts", 0) > prev.get("attempts", 0) or act.get("percent", 0) > 99:
            return True
        return False

    def _state_to_vector(self, s):
        return np.array([
            s["percent"] / 100.0,
            s["attempts"],
            float(s["vehicle"]),
            float(s["is_upside_down"]),
        ], dtype=np.float32)

    def reset(self, timeout=5):
        if self.holding:
            keyboard.release('space')
            self.holding = False

        self.current_vehicle = None
        self.buffer.clear()
        self.prev_state = None
        self.max_percent = 0.0
        self.last_percent = -1.0
        self.last_percent_time = 0.0
        self.percent_frozen = False

        t0 = time.time()
        last_attempt = -1
        stable_since = None
        while time.time() - t0 < timeout:
            s = self.reader.read()
            if s["attempts"] != last_attempt:
                last_attempt = s["attempts"]
                stable_since = time.time()
            elif stable_since is not None and time.time() - stable_since > reset_stable_time and s["percent"] < 0.5:
                break
            time.sleep(0.05)

        frame = self.capture.capture()
        for _ in range(frame_buffer):
            self.buffer.append(frame)

        self.prev_state = self.reader.read()
        self.last_step_time = time.time()
        frames_array = np.stack(self.buffer, axis=0)
        states_vec = self._state_to_vector(self.prev_state)
        return frames_array, states_vec

    def obs(self, new_frame):
        raw_state = self.reader.read()
        new_vehicle = raw_state["vehicle"]
        vehicle_change = (self.current_vehicle is not None and new_vehicle != self.current_vehicle)
        self.current_vehicle = new_vehicle

        if vehicle_change or len(self.buffer) == 0:
            self.buffer.clear()
            for _ in range(frame_buffer):
                self.buffer.append(new_frame)
        else:
            self.buffer.append(new_frame)

        frames_array = np.stack(self.buffer, axis=0)
        state_vector = self._state_to_vector(raw_state)
        return frames_array, state_vector, raw_state

    def step(self, action):
        elapsed = time.time() - self.last_step_time
        sleep = self.step_dt - elapsed
        if sleep > 0:
            time.sleep(sleep)
        self.last_step_time = time.time()

        self.send_input(action)
        frame = self.capture.capture()
        frames_array, vec_state, act_state = self.obs(frame)

        ahora = time.time()
        freeze_just_detected = False

        if act_state["percent"] != self.last_percent:
            self.last_percent = act_state["percent"]
            self.last_percent_time = ahora
            self.percent_frozen = False
        elif (
            not self.percent_frozen
            and act_state["percent"] > 0
            and ahora - self.last_percent_time > self.freeze_threshold
        ):
            self.percent_frozen = True
            freeze_just_detected = True

        reward = self.reward_function(self.prev_state, act_state)

        if freeze_just_detected:
            reward = penalty_death

        done = self.is_done(self.prev_state, act_state)
        self.max_percent = max(self.max_percent, act_state["percent"])
        self.prev_state = act_state

        info = {
            "percent": act_state["percent"],
            "attempts": act_state["attempts"],
            "max_percent": self.max_percent,
            "holding": self.holding,
            "vehicle": act_state["vehicle"],
            "current_vehicle": self.current_vehicle,
            "percent_frozen": self.percent_frozen,
        }
        return frames_array, vec_state, reward, done, info
