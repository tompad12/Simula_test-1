from capturar_juego import GeometryDashCapture
from gd_lector_estado import GDStateReader
from collections import deque
import time
import numpy as np
import keyboard


fps=60
frame_buffer=4
reward_1=0.1 #por sobrevivir
checkpoint_reward=1 #por pasar múltiplos de step_checkpoint
step_checkpoint=3 
reward_100=100 #por pasar el nivel
penalty_death=-10 #por morir
hold_penalty=-0.05   #por mantener espacio
reset_timeout=5.0      
reset_stable_time = 0.3     
reset_max_time = 0.5  

freeze_time = 0.08

class GD_env:
    def __init__(self):
        self.capture=GeometryDashCapture()
        self.reader=GDStateReader()
        self.buffer=deque(maxlen=frame_buffer)
        self.holding=False
        self.last_step_time=0
        self.step_dt=1/fps
        self.prev_state=None
        self.max_percent=0
        self.t_avance = 0.0
    def close(self):
        if self.holding:
            keyboard.release('space')
            self.holding=False
        self.capture.close()
        self.reader.close()
    def send_input(self, action):
        if action==1 and not self.holding:
            keyboard.press('space')
            self.holding=True
        elif action==0 and self.holding:
            keyboard.release('space')
            self.holding=False
    def reward_function(self,prev,act,choque=False):
        reward=reward_1
        prev_checkpoint=int(prev["percent"]/step_checkpoint)
        act_checkpoint=int(act["percent"]/step_checkpoint)
        if act_checkpoint>prev_checkpoint:
            reward=checkpoint_reward
        if act["attempts"]>prev["attempts"] or choque:
            reward=penalty_death
        if act["percent"]>99:
            reward+=100
        if self.holding:
            reward+=hold_penalty
        return reward
    def is_done(self,prev,act, choque=False):
        return act["attempts"]>prev["attempts"] or act["percent"]>99 or choque
    def _state_to_vector(self, s):
        return np.array([
            s["percent"] / 100.0,
            0.0,
            float(s["vehicle"]),
            float(s["is_upside_down"]),
        ], dtype=np.float32)
    
    def reset(self,timeout=5):
        if self.holding:
            keyboard.release('space')
            self.holding=False
        self.buffer.clear()
        self.prev_state=None
        self.max_percent=0.0
        t0=time.time()
        last_attempt=-1
        stable_since=None
        while time.time()-t0<timeout:
            s=self.reader.read()
            if s["attempts"]!=last_attempt:
                last_attempt=s["attempts"]
                stable_since=time.time()
            elif (stable_since is not None and time.time()-stable_since>0.3 and s["percent"]<0.5):
                break
            time.sleep(0.05)

        frame=self.capture.capture()
        for _ in range(frame_buffer):
            self.buffer.append(frame)
        self.prev_state=self.reader.read()
        self.t_avance = time.time()
        self.last_step_time=time.time()
        frames_array=np.stack(self.buffer,axis=0)
        states_vec=self._state_to_vector(self.prev_state)
        return frames_array,states_vec
    def obs(self,new_frame):
        if len(self.buffer)==0:
            for _ in range(frame_buffer):
                self.buffer.append(new_frame)
        else:
            self.buffer.append(new_frame)
        frames_array=np.stack(self.buffer,axis=0)
        raw_state=self.reader.read()
        vec_state=self._state_to_vector(raw_state)
        return frames_array,vec_state,raw_state
    def step(self,action):
        elapsed=time.time()-self.last_step_time
        sleep=self.step_dt -elapsed
        if sleep>0:
            time.sleep(sleep)
        self.last_step_time=time.time()
        self.send_input(action)
        frame=self.capture.capture()
        frames_array,vec_state,act_state=self.obs(frame)
        
        ahora=time.time()
        if act_state["percent"]>self.prev_state["percent"]:
            self.t_avance=ahora
        choque=(act_state["percent"]>0 and ahora-self.t_avance>freeze_time)
        reward=self.reward_function(self.prev_state,act_state,choque)
        done=self.is_done(self.prev_state,act_state,choque)

        self.max_percent=max(self.max_percent,act_state["percent"])
        self.prev_state=act_state
        info = {"percent": act_state["percent"],"attempts": act_state["attempts"],"max_percent": self.max_percent,"holding": self.holding,}
        return frames_array, vec_state, reward, done, info

