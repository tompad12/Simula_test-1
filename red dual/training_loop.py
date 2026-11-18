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
from replay_buffer import replay_buffer, capacity_buffer
from GD_DQN import GD_DQN
from GD_DQN import crear_redes, update_target

learning_rate=1e-4
gamma=0.99
batch_size=64
capacity_buffer=50000
train_freq=2
eps_start=0.3
eps_end=0.1
decay_steps=500000
target_update_freq=1000
total_steps=1500000
save_freq=20000
log_freq=1000
csv_freq=100
checkpoint_dir="checkpoints"
log_dir="logs"
csv_path=os.path.join(log_dir,"training.csv")
os.makedirs(checkpoint_dir,exist_ok=True)
os.makedirs(log_dir,exist_ok=True)
device="cuda" if torch.cuda.is_available() else "cpu"

def eps_decae(step):
    if step>=decay_steps:
        return eps_end
    progreso=step/decay_steps
    return eps_start+progreso*(eps_end-eps_start)
def choose_accion(online,obs_frame,obs_state,eps,device):
    if random.random()<eps:
        return random.randint(0,1)
    x=torch.from_numpy(obs_frame).float().unsqueeze(0)/255
    s=torch.from_numpy(obs_state).float().unsqueeze(0)
    x=x.to(device)
    s=s.to(device)
    with torch.no_grad():
        q_vals=online(x,s)
        accion=q_vals.argmax(dim=1).item()
    return accion
def prep_batch(batch,device):
    frames,states,actions,rewards,next_frames,next_states,dones=batch

    frames=torch.from_numpy(frames).float().to(device)/255
    next_frames=torch.from_numpy(next_frames).float().to(device)/255
    states=torch.from_numpy(states).float().to(device)
    actions=torch.from_numpy(actions).long().to(device)
    rewards=torch.from_numpy(rewards).float().to(device)
    next_states=torch.from_numpy(next_states).float().to(device)
    dones=torch.from_numpy(dones).float().to(device)
    return frames,states,actions,rewards,next_frames,next_states,dones
def loss(online,target,batch,gamma,device):
    #"I    |  I i"
    #" I I |  I -"
    #"is this loss?(yes)"
    frames,states,actions,rewards,next_frames,next_states,dones=batch
    q_vals=online(frames,states)
    q_act=q_vals.gather(1,actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next_online=online(next_frames,next_states)
        best_action=q_next_online.argmax(dim=1,keepdim=True)
        q_next_target=target(next_frames,next_states)
        q_next_best=q_next_target.gather(1,best_action).squeeze(1)
        target_vals=rewards+gamma*q_next_best*(1-dones)
    loss = nn.SmoothL1Loss()(q_act, target_vals)
    return loss
def paso_opt(optimizer,loss_val,online):
    optimizer.zero_grad()
    loss_val.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(),max_norm=10)
    optimizer.step()

def guardar_checkpoint(ruta, online, target, optimizer,step, episode, epsilon, best):
    torch.save({
        "online_state":online.state_dict(),
        "target_state":target.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "step":step,
        "episode":episode,
        "epsilon":epsilon,
        "best":best,}, ruta)

    
def cargar_checkpoint(ruta, online, target, optimizer):
    checkpoint=torch.load(ruta,map_location=device,weights_only=False)
    online.load_state_dict(checkpoint["online_state"])
    target.load_state_dict(checkpoint["target_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (checkpoint["step"],checkpoint["episode"],checkpoint["epsilon"],checkpoint["best"])

def escribir_csv(step, episode, epsilon, loss, reward, max_percent, buf_len,ruta=csv_path):
    archivo_nuevo = not os.path.exists(ruta)

    with open(ruta,"a",newline="") as f:
        writer=csv.writer(f)
        if archivo_nuevo:
            writer.writerow(["step", "episode", "epsilon", "loss","reward_avg", "max_percent", "buffer_size"])
        writer.writerow([step, episode, f"{epsilon:.4f}", f"{loss:.4f}",f"{reward:.4f}", f"{max_percent:.2f}", buf_len])

#optimizer = optim.Adam(online.parameters(), lr=learning_rate)
def main():
    env=GD_env()
    buf=replay_buffer(cap=capacity_buffer)
    online, target=crear_redes(device)
    optimizer=optim.Adam(online.parameters(),lr=learning_rate)
    resume_path="checkpoints/best.pt"
    if resume_path and os.path.exists(resume_path):
        print(f"Pesos antiguos detectados, reanudando desde {resume_path}")
        step_inicial,episode,epsilon_inicial,best_max=\
            cargar_checkpoint(resume_path,online,target,optimizer)
    else:
        step_inicial=1
        episode=0
        epsilon_inicial=eps_start
        best_max=0

    obs_frames,obs_states=env.reset()
    
    episode_reward=0
    episode_steps=0
    episode_max_percent=0
    losses_recientes=deque(maxlen=100)
    rewards_recientes=deque(maxlen=100)
    for step in range(step_inicial,total_steps+1):
        epsilon=eps_decae(step)
        action=choose_accion(online,obs_frames,obs_states,epsilon,device)
        next_frames,next_state,reward,done,info=env.step(action)
        buf.push(obs_frames,obs_states,action,reward,next_frames,next_state,done)
        episode_reward+=reward
        episode_steps+=1
        episode_max_percent=max(episode_max_percent,info["percent"])
        if step % train_freq==0 and len(buf)>=batch_size:
            batch=buf.sample(batch_size)
            batch_gpu=prep_batch(batch,device)
            loss_valor=loss(online,target,batch_gpu,gamma,device)
            paso_opt(optimizer,loss_valor,online)
            losses_recientes.append(loss_valor.item())
        if step%target_update_freq==0:
            update_target(online,target)
        if done:
            rewards_recientes.append(episode_reward)
            if episode_max_percent>best_max:
                best_max=episode_max_percent
                guardar_checkpoint(
                    os.path.join(checkpoint_dir, "best.pt"),
                    online, target, optimizer, step, episode,
                    epsilon, best_max)
                print(f"nuevo récord: {best_max:.1f}%")
            episode+=1
            episode_reward=0
            episode_steps=0
            episode_max_percent=0
            obs_frames,obs_states=env.reset()
        else:
            obs_frames,obs_states=next_frames,next_state
        if step%csv_freq==0:
            escribir_csv(step, episode, epsilon,
                         np.mean(losses_recientes) if losses_recientes else 0.0,
                         np.mean(rewards_recientes) if rewards_recientes else 0.0,
                         episode_max_percent, len(buf))
        if step % save_freq == 0:
            ruta = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
            guardar_checkpoint(ruta, online, target, optimizer,
                               step, episode, epsilon, best_max)

    print("-" * 60)
    print(f"Entrenamiento terminado. Best max%: {best_max:.1f}")
    guardar_checkpoint(
        os.path.join(checkpoint_dir, "final.pt"),
        online, target, optimizer, total_steps, episode,
        eps_decae(total_steps), best_max)
    env.close()

if __name__ == "__main__":
    main()



