import os
import time
import numpy as np
import keyboard
from collections import deque
from capturar_juego import GeometryDashCapture
from gd_lector_estado import GDStateReader
from gd_env import GD_env   # solo para reutilizar su _state_to_vector (no se crea ningún entorno)

fps = 37                  # mismo ritmo aproximado con el que actúa el agente (antes: 60)
frame_buffer = 4
DURACION_SEG = 600        # 10 minutos
SALIDA = "demos/human_play_01.npz"   # cambia el nombre en cada sesión para no pisar archivos
TECLA_PAUSA = "c"

def state_to_vector(s):
    # Usa EXACTAMENTE la misma función que gd_env.py. Si algún día cambian el vector de
    # estado en el entorno (por ejemplo, poner attempts en 0.0), las demos lo siguen solas.
    return GD_env._state_to_vector(None, s)

def main():
    time.sleep(1)
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)   # evita perder la grabación al final

    capture = GeometryDashCapture()
    reader = GDStateReader()
    frames_buf = deque(maxlen=frame_buffer)

    all_frames = []
    all_states = []
    all_actions = []
    all_attempts = []   # attempts crudo, solo para separar episodios después

    step_dt = 1.0 / fps
    t0 = time.time()
    last_step = time.time()

    print("Grabando. Juega GD con la BARRA ESPACIADORA (no con el mouse). Ctrl+C para parar.")


    pausado = [False]
    def alternar():
        pausado[0] = not pausado[0]
        print("PAUSA" if pausado[0] else "GRABANDO")
    keyboard.add_hotkey(TECLA_PAUSA, alternar)
    try:
        while time.time() - t0 < DURACION_SEG:
            if pausado[0]:
                frames_buf.clear()        # al reanudar, el apilado de 4 frames parte limpio
                last_step = time.time()
                time.sleep(0.05)
                continue
            elapsed = time.time() - last_step
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)
            last_step = time.time()

            frame = capture.capture()
            state = reader.read()
            action = 1 if keyboard.is_pressed("space") else 0

            if len(frames_buf) == 0:
                for _ in range(frame_buffer):
                    frames_buf.append(frame)
            else:
                frames_buf.append(frame)
            frames_array = np.stack(frames_buf, axis=0)

            all_frames.append(frames_array.copy())
            all_states.append(state_to_vector(state))
            all_actions.append(action)
            all_attempts.append(state["attempts"])

            if len(all_frames) % 600 == 0:
                print(f"Grabados {len(all_frames)} steps ({len(all_frames)/fps:.1f}s)")

    except KeyboardInterrupt:
        print("Grabación terminada.")
    finally:
        capture.close()
        reader.close()

    np.savez_compressed(
        SALIDA,
        frames=np.stack(all_frames),                      # (N, 4, 84, 84) uint8
        states=np.stack(all_states),                      # (N, 4) float32
        actions=np.array(all_actions, dtype=np.int64),    # (N,)
        attempts=np.array(all_attempts, dtype=np.int64),  # (N,)
    )
    print(f"Guardado en {SALIDA}: {len(all_frames)} transiciones")

if __name__ == "__main__":
    main()