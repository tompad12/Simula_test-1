import time
import numpy as np
import keyboard
from collections import deque
from capturar_juego import GeometryDashCapture
from gd_lector_estado import GDStateReader

fps = 60
frame_buffer = 4
DURACION_SEG = 1800   # 10 minutos

def state_to_vector(s):
    return np.array([
        s["percent"] / 100.0,
        s["attempts"],
        float(s["vehicle"]),
        float(s["is_upside_down"]),
    ], dtype=np.float32)

def main():
    capture = GeometryDashCapture()
    reader = GDStateReader()
    frames_buf = deque(maxlen=frame_buffer)

    all_frames = []   # listas de numpy arrays
    all_states = []
    all_actions = []

    step_dt = 1.0 / fps
    t0 = time.time()
    last_step = time.time()

    print("Grabando. Juega GD. Ctrl+C para parar.")

    try:
        while time.time() - t0 < DURACION_SEG:
            # Rate limit
            elapsed = time.time() - last_step
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)
            last_step = time.time()

            # Capturar frame y estado
            frame = capture.capture()
            state = reader.read()
            holding = keyboard.is_pressed("space")
            action = 1 if holding else 0

            # Stack
            if len(frames_buf) == 0:
                for _ in range(frame_buffer):
                    frames_buf.append(frame)
            else:
                frames_buf.append(frame)
            frames_array = np.stack(frames_buf, axis=0)

            all_frames.append(frames_array.copy())
            all_states.append(state_to_vector(state))
            all_actions.append(action)

            if len(all_frames) % 600 == 0:
                print(f"Grabados {len(all_frames)} steps "
                      f"({len(all_frames)/fps:.1f}s)")

    except KeyboardInterrupt:
        print("Grabación terminada.")
    finally:
        capture.close()
        reader.close()

    # Guardar
    frames_np = np.stack(all_frames)     # (N, 4, 84, 84) uint8
    states_np = np.stack(all_states)     # (N, 4) float32
    actions_np = np.array(all_actions)   # (N,) int64

    np.savez_compressed(
        "demos/human_play_02.npz",
        frames=frames_np,
        states=states_np,
        actions=actions_np,
    )
    print(f"Guardado: {len(all_frames)} transiciones")

if __name__ == "__main__":
    main()