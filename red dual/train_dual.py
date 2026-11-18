import os
import csv
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gd_env import GD_env
from replay_buffer import replay_buffer, capacity_buffer
from GD_DQN import GD_DQN, update_target
from training_loop import choose_accion, prep_batch, loss, paso_opt


# ============================================================
# HIPERPARÁMETROS DEL CUBO
# ============================================================
cube_learning_rate = 1e-4
cube_eps_start     = 0.30      # si reanudas del 25%, empieza bajo
cube_eps_end       = 0.05
cube_decay_steps   = 500_000
cube_batch_size    = 64
cube_train_freq    = 2
cube_target_update = 1000

# ============================================================
# HIPERPARÁMETROS DE LA NAVE
# ============================================================
ship_learning_rate = 1e-4
ship_eps_start     = 0.5       # nave entrena desde cero → exploración total
ship_eps_end       = 0.05
ship_decay_steps   = 500_000
ship_batch_size    = 64
ship_train_freq    = 2
ship_target_update = 1000

# ============================================================
# COMPARTIDOS
# ============================================================
gamma       = 0.99
total_steps = 1_000_000+500000
save_freq   = 20_000
log_freq    = 1_000
csv_freq    = 100

# --- Trackers de frontera (global, no por episodio) ---
cube_max_ever = 0.0
ship_max_ever = 0.0

FRONTIER_WINDOW = 3.0    # % — a cuántos puntos del récord se considera "frontera"
FRONTIER_EPS    = 0.15   # ε mínimo en la frontera

# Rutas
cube_resume_path = "checkpoints/best.pt"   # opcional: cargar el 25%
checkpoint_dir   = "checkpoints/dual"
log_dir          = "logs"
csv_path         = os.path.join(log_dir, "dual.csv")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def cube_eps_decae(step):
    if step >= cube_decay_steps:
        return cube_eps_end
    progreso = step / cube_decay_steps
    return cube_eps_start + progreso * (cube_eps_end - cube_eps_start)


def ship_eps_decae(step):
    if step >= ship_decay_steps:
        return ship_eps_end
    progreso = step / ship_decay_steps
    return ship_eps_start + progreso * (ship_eps_end - ship_eps_start)


def main():
    # --- Modelos (4 en total) ---
    cube_online = GD_DQN().to(device)
    cube_target = GD_DQN().to(device)
    ship_online = GD_DQN().to(device)
    ship_target = GD_DQN().to(device)

    # Inicializar LazyLinear
    dummy_x = torch.zeros(1, 4, 84, 84, device=device)
    dummy_s = torch.zeros(1, 4, device=device)
    for m in [cube_online, cube_target, ship_online, ship_target]:
        m(dummy_x, dummy_s)

    # Cargar cubo desde checkpoint (opcional)
    BC_PRETRAINED_PATH =  None#"checkpoints/bc_pretrained.pt"
    DUAL_RESUME_PATH   = "checkpoints/dual/final_dual.pt"   # o el que quieras
    CUBE_RESUME_PATH   = "checkpoints/best.pt"

    if BC_PRETRAINED_PATH is not None and os.path.exists(BC_PRETRAINED_PATH):
        # --- CASO A: empezar desde BC preentrenado ---
        print(f"Cargando pesos BC desde {BC_PRETRAINED_PATH}")
        bc_ckpt = torch.load(BC_PRETRAINED_PATH,
                             map_location=device, weights_only=False)

        for m in [cube_online, cube_target, ship_online, ship_target]:
            m.load_state_dict(bc_ckpt["online_state"])

        # lr reducido para no destruir lo aprendido
        opt_cube = optim.Adam(cube_online.parameters(), lr=cube_learning_rate * 0.5)
        opt_ship = optim.Adam(ship_online.parameters(), lr=ship_learning_rate * 0.5)

        step_inicial = 1
        cube_steps   = 0
        ship_steps   = 0
        episode      = 0
        best_cube    = 0.0
        best_ship    = 0.0

    elif DUAL_RESUME_PATH is not None and os.path.exists(DUAL_RESUME_PATH):
        # --- CASO B: reanudar desde checkpoint dual ---
        print(f"Reanudando desde checkpoint dual: {DUAL_RESUME_PATH}")
        ckpt = torch.load(DUAL_RESUME_PATH, map_location=device, weights_only=False)

        cube_online.load_state_dict(ckpt["cube_online"])
        cube_target.load_state_dict(ckpt["cube_target"])
        ship_online.load_state_dict(ckpt["ship_online"])
        ship_target.load_state_dict(ckpt["ship_target"])

        opt_cube = optim.Adam(cube_online.parameters(), lr=cube_learning_rate)
        opt_ship = optim.Adam(ship_online.parameters(), lr=ship_learning_rate)
        opt_cube.load_state_dict(ckpt["opt_cube"])
        opt_ship.load_state_dict(ckpt["opt_ship"])

        step_inicial = ckpt["step"]
        cube_steps   = ckpt["cube_steps"]
        ship_steps   = ckpt["ship_steps"]
        episode      = ckpt["episode"]
        best_cube    = ckpt["best_cube"]
        best_ship    = ckpt["best_ship"]
        print(f"  step={step_inicial} | cube_steps={cube_steps} "
              f"| ship_steps={ship_steps}")
        print(f"  best_cube={best_cube:.1f}% | best_ship={best_ship:.1f}%")

    elif os.path.exists(CUBE_RESUME_PATH):
        # --- CASO C: empezar dual con cubo preentrenado (single-model) ---
        print(f"Cubo cargado desde {CUBE_RESUME_PATH}")
        ckpt = torch.load(CUBE_RESUME_PATH, map_location=device, weights_only=False)
        cube_online.load_state_dict(ckpt["online_state"])
        cube_target.load_state_dict(ckpt["target_state"])
        print(f"  best del cubo: {ckpt['best']:.1f}%")
        ship_target.load_state_dict(ship_online.state_dict())

        opt_cube = optim.Adam(cube_online.parameters(), lr=cube_learning_rate)
        opt_ship = optim.Adam(ship_online.parameters(), lr=ship_learning_rate)

        step_inicial = 1
        cube_steps   = 0
        ship_steps   = 0
        episode      = 0
        best_cube    = ckpt["best"]
        best_ship    = 0.0

    else:
        # --- CASO D: empezar desde cero ---
        print("Empezando dual desde cero")
        cube_target.load_state_dict(cube_online.state_dict())
        ship_target.load_state_dict(ship_online.state_dict())

        opt_cube = optim.Adam(cube_online.parameters(), lr=cube_learning_rate)
        opt_ship = optim.Adam(ship_online.parameters(), lr=ship_learning_rate)

        step_inicial = 1
        cube_steps   = 0
        ship_steps   = 0
        episode      = 0
        best_cube    = 0.0
        best_ship    = 0.0

    # Congelar targets (importante tras cargar)
    for p in cube_target.parameters():
        p.requires_grad = False
    for p in ship_target.parameters():
        p.requires_grad = False

    print(f"  lr_cube={opt_cube.param_groups[0]['lr']} "
          f"lr_ship={opt_ship.param_groups[0]['lr']}")


    # --- Buffers ---
    buf_cube = replay_buffer(cap=capacity_buffer)
    buf_ship = replay_buffer(cap=capacity_buffer)

    # --- Entorno ---
    env = GD_env()
    obs_frames, obs_state = env.reset()

    # --- Estado del episodio ---
    episode = 0
    ep_reward_cube = 0.0
    ep_reward_ship = 0.0
    ep_max_pct_cube = 0.0
    ep_max_pct_ship = 0.0
    best_cube = 0.0
    best_ship = 0.0

    cube_max_ever = 0.0
    ship_max_ever = 0.0
    # --- Contadores por vehículo ---
    cube_steps = 0
    ship_steps = 0

    # --- Métricas ---
    loss_cube_recientes = deque(maxlen=100)
    loss_ship_recientes = deque(maxlen=100)
    reward_cube_recientes = deque(maxlen=100)
    reward_ship_recientes = deque(maxlen=100)

    # --- CSV ---
    archivo_nuevo = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if archivo_nuevo:
        csv_writer.writerow([
            "step", "cube_steps", "ship_steps", "episode",
            "eps_cube", "eps_ship", "loss_cube", "loss_ship",
            "reward_cube_avg", "reward_ship_avg",
            "max_pct_cube", "max_pct_ship",
            "buf_cube_size", "buf_ship_size",
        ])

    print(f"Device: {device} | total_steps: {total_steps}")
    print("-" * 60)

    step = 0
    try:
        for step in range(step_inicial, total_steps + 1):
            vehicle = env.prev_state["vehicle"]

            # --- Elegir acción según vehículo ---
            if vehicle == env.VEHICLE_CUBE:
                pct = env.prev_state["percent"]
                if pct > cube_max_ever:
                    cube_max_ever = pct
                base_eps = cube_eps_decae(cube_steps)
                if cube_max_ever > 5 and pct > cube_max_ever - FRONTIER_WINDOW:
                    eps = max(base_eps, FRONTIER_EPS)
                else:
                    eps = base_eps
                action = choose_accion(cube_online, obs_frames, obs_state, eps, device)
            elif vehicle == env.VEHICLE_SHIP:
                pct = env.prev_state["percent"]
                if pct > ship_max_ever:
                    ship_max_ever = pct
                base_eps = ship_eps_decae(ship_steps)
                if ship_max_ever > 30 and pct > ship_max_ever - FRONTIER_WINDOW:
                    eps = max(base_eps, FRONTIER_EPS)
                else:
                    eps = base_eps

                action = choose_accion(ship_online, obs_frames, obs_state, eps, device)

            else:
                action = 0
            next_frames, next_state, reward, done, info = env.step(action)

            # --- Guardar + entrenar según vehículo ---
            if vehicle == env.VEHICLE_CUBE:
                buf_cube.push(obs_frames, obs_state, action, reward,
                              next_frames, next_state, done)
                cube_steps += 1
                ep_reward_cube += reward
                ep_max_pct_cube = max(ep_max_pct_cube, info["percent"])

                if cube_steps % cube_train_freq == 0 and len(buf_cube) >= cube_batch_size:
                    batch = buf_cube.sample(cube_batch_size)
                    batch_gpu = prep_batch(batch, device)
                    loss_val = loss(cube_online, cube_target, batch_gpu, gamma, device)
                    paso_opt(opt_cube, loss_val, cube_online)
                    loss_cube_recientes.append(loss_val.item())

                if cube_steps % cube_target_update == 0:
                    update_target(cube_online, cube_target)

            elif vehicle == env.VEHICLE_SHIP:
                buf_ship.push(obs_frames, obs_state, action, reward,
                              next_frames, next_state, done)
                ship_steps += 1
                ep_reward_ship += reward
                ep_max_pct_ship = max(ep_max_pct_ship, info["percent"])

                if ship_steps % ship_train_freq == 0 and len(buf_ship) >= ship_batch_size:
                    batch = buf_ship.sample(ship_batch_size)
                    batch_gpu = prep_batch(batch, device)
                    loss_val = loss(ship_online, ship_target, batch_gpu, gamma, device)
                    paso_opt(opt_ship, loss_val, ship_online)
                    loss_ship_recientes.append(loss_val.item())

                if ship_steps % ship_target_update == 0:
                    update_target(ship_online, ship_target)

            # --- Fin de episodio ---
            if done:
                if ep_max_pct_cube > 0:
                    reward_cube_recientes.append(ep_reward_cube)
                    if ep_max_pct_cube > best_cube:
                        best_cube = ep_max_pct_cube
                        torch.save({
                            "online_state": cube_online.state_dict(),
                            "target_state": cube_target.state_dict(),
                            "optimizer":    opt_cube.state_dict(),
                            "step": step, "cube_steps": cube_steps,
                            "episode": episode, "best": best_cube,
                        }, os.path.join(checkpoint_dir, "best_cube.pt"))
                        print(f"⭐ nuevo récord cubo: {best_cube:.1f}%")

                if ep_max_pct_ship > 0:
                    reward_ship_recientes.append(ep_reward_ship)
                    if ep_max_pct_ship > best_ship:
                        best_ship = ep_max_pct_ship
                        torch.save({
                            "online_state": ship_online.state_dict(),
                            "target_state": ship_target.state_dict(),
                            "optimizer":    opt_ship.state_dict(),
                            "step": step, "ship_steps": ship_steps,
                            "episode": episode, "best": best_ship,
                        }, os.path.join(checkpoint_dir, "best_ship.pt"))
                        print(f"⭐ nuevo récord nave: {best_ship:.1f}%")

                episode += 1
                ep_reward_cube = 0.0
                ep_reward_ship = 0.0
                ep_max_pct_cube = 0.0
                ep_max_pct_ship = 0.0
                obs_frames, obs_state = env.reset()
            else:
                obs_frames, obs_state = next_frames, next_state

            # --- Logs ---
            if step % log_freq == 0:
                lc = np.mean(loss_cube_recientes) if loss_cube_recientes else 0.0
                ls = np.mean(loss_ship_recientes) if loss_ship_recientes else 0.0
                print(f"[step {step:7d}] "
                      f"cube_steps={cube_steps:6d} ship_steps={ship_steps:6d} | "
                      f"ε_c={cube_eps_decae(cube_steps):.3f} "
                      f"ε_s={ship_eps_decae(ship_steps):.3f} | "
                      f"loss_c={lc:.4f} loss_s={ls:.4f} | "
                      f"best_c={best_cube:.1f}% best_s={best_ship:.1f}%")

            if step % csv_freq == 0:
                csv_writer.writerow([
                    step, cube_steps, ship_steps, episode,
                    f"{cube_eps_decae(cube_steps):.4f}",
                    f"{ship_eps_decae(ship_steps):.4f}",
                    f"{np.mean(loss_cube_recientes) if loss_cube_recientes else 0.0:.4f}",
                    f"{np.mean(loss_ship_recientes) if loss_ship_recientes else 0.0:.4f}",
                    f"{np.mean(reward_cube_recientes) if reward_cube_recientes else 0.0:.4f}",
                    f"{np.mean(reward_ship_recientes) if reward_ship_recientes else 0.0:.4f}",
                    f"{ep_max_pct_cube:.2f}",
                    f"{ep_max_pct_ship:.2f}",
                    len(buf_cube), len(buf_ship),
                ])
                csv_file.flush()

            if step % save_freq == 0:
                torch.save({
                    "cube_online": cube_online.state_dict(),
                    "cube_target": cube_target.state_dict(),
                    "opt_cube":    opt_cube.state_dict(),
                    "ship_online": ship_online.state_dict(),
                    "ship_target": ship_target.state_dict(),
                    "opt_ship":    opt_ship.state_dict(),
                    "step": step, "cube_steps": cube_steps,
                    "ship_steps": ship_steps, "episode": episode,
                    "best_cube": best_cube, "best_ship": best_ship,
                }, os.path.join(checkpoint_dir, f"dual_step_{step}.pt"))

    finally:
        torch.save({
            "cube_online": cube_online.state_dict(),
            "cube_target": cube_target.state_dict(),
            "opt_cube":    opt_cube.state_dict(),
            "ship_online": ship_online.state_dict(),
            "ship_target": ship_target.state_dict(),
            "opt_ship":    opt_ship.state_dict(),
            "step": step, "cube_steps": cube_steps,
            "ship_steps": ship_steps, "episode": episode,
            "best_cube": best_cube, "best_ship": best_ship,
        }, os.path.join(checkpoint_dir, "final_dual.pt"))
        csv_file.close()
        env.close()
        print(f"\nGuardado. best_cube={best_cube:.1f}% best_ship={best_ship:.1f}%")


if __name__ == "__main__":
    main()