import os
import time
import csv
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gd_env import GD_env
from replay_buffer import replay_buffer, capacity_buffer, size_batch
from GD_DQN import GD_DQN
from GD_DQN import crear_redes_dual, cargar_modelo_congelado, update_target
from training_loop import choose_accion, eps_decae, prep_batch, loss, paso_opt

learning_rate=1e-4
gamma=0.99
batch_size=size_batch
train_freq=2
target_update_freq=1000
ship_eps_start=0.8
ship_eps_end=0.001
ship_eps_decay_steps=6000
ship_checkpoint_step=1
total_steps=200000
save_freq=20000
log_freq=1000
csv_freq=100
cube_model_path="checkpoints/best.pt"
checkpoint_dir="checkpoints/ship"
log_dir="logs"
csv_path=os.path.join(log_dir, "ship.csv")
os.makedirs(checkpoint_dir,exist_ok=True)
os.makedirs(log_dir,exist_ok=True)
device="cuda" if torch.cuda.is_available() else "cpu"



def ship_eps_decae(step):
    if step>=ship_eps_decay_steps:
        return ship_eps_end
    progreso=step/ship_eps_decay_steps
    return ship_eps_start+progreso*(ship_eps_end-ship_eps_start)


def setup_infraestructura():
    cube_online,cube_target,ship_online,ship_target=crear_redes_dual(device)
    cargar_modelo_congelado(cube_online,cube_model_path,device)
    cube_target.load_state_dict(cube_online.state_dict())
    for p in cube_target.parameters():
        p.requires_grad=False
    buf_cube=replay_buffer(cap=capacity_buffer)
    buf_ship=replay_buffer(cap=capacity_buffer)
    opt_ship=optim.Adam(ship_online.parameters(),lr=learning_rate)
    return {"cube_online":cube_online,"cube_target":cube_target,"ship_online":ship_online,"ship_target":ship_target,"buf_cube":buf_cube,"buf_ship":buf_ship,"opt_ship":opt_ship}

def main():
    infra=setup_infraestructura()
    cube_online=infra["cube_online"]
    cube_target=infra["cube_target"]
    ship_online=infra["ship_online"]
    ship_target=infra["ship_target"]
    buf_ship=infra["buf_ship"]
    opt_ship=infra["opt_ship"]
    env=GD_env()
    obs_frames,obs_state=env.reset()
    episode=0
    episode_reward_ship=0
    episode_ship_steps=0
    episode_max_percent_ship=0
    best_ship_percent=0
    ship_steps=0
    loss_recientes=deque(maxlen=100)
    reward_recientes=deque(maxlen=100)
    archivo_nuevo=not os.path.exists(csv_path)
    csv_file=open(csv_path,"a",newline="")
    csv_writer=csv.writer(csv_file)
    if archivo_nuevo:
        csv_writer.writerow(["step","ship_steps","episode","epsilon_ship","loss","reward_ship","max_percent_ship","buf_ship_size"])
    step=0
    print(f"Empezando a entrenar. Pasos totales ={total_steps}")
    try:
        for step in range(1,total_steps+1):
            vehicle=env.prev_state["vehicle"]
            if vehicle==env.VEHICLE_CUBE:                
                action=choose_accion(cube_online,obs_frames,obs_state,eps=0,device=device)
            elif vehicle==env.VEHICLE_SHIP:
                eps_ship=ship_eps_decae(ship_steps)
                action=choose_accion(cube_online,obs_frames,obs_state,eps=eps_ship,device=device)
            else: 
                action=0
            next_frames,next_state,reward,done,info=env.step(action)
            if vehicle==env.VEHICLE_SHIP:
                buf_ship.push(obs_frames,obs_state,action,reward,next_frames,next_state,done)
                ship_steps+=1
                episode_reward_ship+=reward
                episode_ship_steps+=1
                episode_max_percent_ship=max(episode_max_percent_ship,info["percent"])
                if ship_steps%train_freq==0 and len(buf_ship)>=batch_size:
                    batch=buf_ship.sample(batch_size)
                    batch_gpu=prep_batch(batch,device)
                    loss_val=loss(ship_online,ship_target,batch_gpu,gamma,device)
                    paso_opt(opt_ship,loss_val,ship_online)
                    loss_recientes.append(loss_val.item())
                if ship_steps%target_update_freq==0:
                    update_target(ship_online,ship_target)
            if done:
                if episode_ship_steps>0:
                    reward_recientes.append(episode_reward_ship)
                    if episode_max_percent_ship>best_ship_percent:
                        best_ship_percent=episode_max_percent_ship
                        torch.save({"ship_online": ship_online.state_dict(),"ship_target": ship_target.state_dict(),"opt_ship": opt_ship.state_dict(),"step": step,"ship_steps":  ship_steps,"episode": episode,"best_ship":best_ship_percent}, os.path.join(checkpoint_dir, "best_ship.pt"))
                episode+=1
                episode_reward_ship=0
                episode_ship_steps=0
                episode_max_percent_ship=0
                obs_frames,obs_state=env.reset()
            else:
                obs_frames,obs_state=next_frames,next_state
            #==========logs================
            if step % log_freq == 0:
                    loss_avg = np.mean(loss_recientes) if loss_recientes else 0.0
                    reward_avg = np.mean(reward_recientes) if reward_recientes else 0.0
                    print(f"[step {step:7d}] ship_steps={ship_steps:6d} "
                        f"buf_ship={len(buf_ship):6d} "
                        f"ε_ship={ship_eps_decae(ship_steps):.3f} "
                        f"loss={loss_avg:.4f} "
                        f"R_ship_avg={reward_avg:+.2f} "
                        f"best_ship%={best_ship_percent:.1f}")

            if step % csv_freq == 0:
                csv_writer.writerow([
                    step, ship_steps, episode,
                    f"{ship_eps_decae(ship_steps):.4f}",
                    f"{np.mean(loss_recientes) if loss_recientes else 0.0:.4f}",
                    f"{np.mean(reward_recientes) if reward_recientes else 0.0:.4f}",
                    f"{episode_max_percent_ship:.2f}",
                    len(buf_ship),
                    ])
                csv_file.flush()

            if step % save_freq == 0:
                ruta = os.path.join(checkpoint_dir, f"ship_step_{step}.pt")
                torch.save({
                    "ship_online": ship_online.state_dict(),
                    "ship_target": ship_target.state_dict(),
                    "opt_ship":    opt_ship.state_dict(),
                    "step":        step,
                    "ship_steps":  ship_steps,
                    "episode":     episode,
                    "best_ship":   best_ship_percent,
                }, ruta)
            #==========
    finally:
        torch.save({
            "ship_online": ship_online.state_dict(),
            "ship_target": ship_target.state_dict(),
            "opt_ship":    opt_ship.state_dict(),
            "step":        step,
            "ship_steps":  ship_steps,
            "episode":     episode,
            "best_ship":   best_ship_percent,
        }, os.path.join(checkpoint_dir, "final_ship.pt"))
        csv_file.close()
        env.close()

        



if __name__ == "__main__":
    main() 