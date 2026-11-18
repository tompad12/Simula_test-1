import csv
import os
import sys

import numpy as np
import torch

from gd_env import GD_env
from GD_DQN import crear_redes, device

# ---- Edita esto y presiona Run, o pasa argumentos por la terminal ----
CHECKPOINTS = [
    "checkpoints_lr/checkpoint_step_1040000.pt",
    "checkpoints_lr/checkpoint_step_900000.pt",
    "checkpoints_lr/promedio_820_900.pt",
]
N_EPISODIOS = 90
# En el entrenamiento el ritmo promedio fue ~37 steps/s (el juego no se pausa
# mientras la red entrena). Se imita aquí para que la política vea el mismo ritmo.
# Si algún día entrenas con timing fijo, pon None.
PASO_EVAL = 1 / 37
# ----------------------------------------------------------------------


def accion_greedy(online, obs_f, obs_s):
    x = torch.from_numpy(obs_f).float().unsqueeze(0).to(device) / 255
    s = torch.from_numpy(obs_s).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return online(x, s).argmax(dim=1).item()


def evaluar(env, ruta, n_episodios):
    online, _ = crear_redes(device)
    ck = torch.load(ruta, map_location=device, weights_only=False)
    online.load_state_dict(ck["online_state"])
    online.eval()

    resultados = []
    for i in range(n_episodios):
        obs_f, obs_s = env.reset()
        max_pct, done = 0.0, False
        while not done:
            accion = accion_greedy(online, obs_f, obs_s)
            obs_f, obs_s, _, done, info = env.step(accion)
            max_pct = max(max_pct, info["percent"])
        resultados.append(max_pct)
        print(f"  episodio {i + 1}/{n_episodios}: {max_pct:.2f}%")
    return np.array(resultados)


def main():
    if len(sys.argv) > 1:
        rutas = [sys.argv[1]]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else N_EPISODIOS
    else:
        rutas, n = CHECKPOINTS, N_EPISODIOS

    for r in rutas:
        if not os.path.exists(r):
            print(f"No encuentro {r}")
            if os.path.isdir("checkpoints"):
                print("En la carpeta checkpoints hay:", sorted(os.listdir("checkpoints")))
            else:
                print("No hay carpeta 'checkpoints' aquí: corre el script desde la carpeta del proyecto.")
            return

    env = GD_env()
    if PASO_EVAL:
        env.step_dt = PASO_EVAL
    nuevo = not os.path.exists("evaluacion.csv")
    try:
        with open("evaluacion.csv", "a", newline="") as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(["checkpoint", "percent_final"])
            for r in rutas:
                print(f"Evaluando {r}")
                res = evaluar(env, r, n)
                for p in res:
                    w.writerow([os.path.basename(r), f"{p:.2f}"])
                f.flush()
                print(f"-> media {res.mean():.2f}% | mediana {np.median(res):.2f}% | "
                      f"máx {res.max():.2f}% | mueren antes del 10%: {(res < 10).mean() * 100:.0f}%")
    finally:
        env.close()


if __name__ == "__main__":
    main()