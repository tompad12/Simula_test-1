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
from replay_buffer_doble import replay_buffer_doble
from replay_buffer_tramos import replay_buffer_tramos
from GD_DQN import GD_DQN
from GD_DQN import crear_redes, update_target
from nstep import NStepAcumulador
from ema_pesos import EMAPesos, decay_desde_ventana 
from grabar_gif import GrabadorGIF

learning_rate=1e-4
lr_end=1e-5
lr_inicio=1000
lr_duracion=1500000
gamma=0.99
N_STEP=3
ema_ventana=100000

resume_path="checkpoints_nstep/interrumpido.pt"
REINICIAR_EPS=True
eps_reinicio=(0.09,0.001,100000)
REINICIAR_LR=True
lr_reinicio=(5e-6,1e-7,400000)

batch_size=64
capacity_buffer=100000
min_buffer = 5000
cap_m = 10000
vm = 5
fm = 0.125
train_freq=2
eps_start=0.2
eps_end=0.001
decay_steps=1200000
target_update_freq=1000
total_steps=8000000
save_freq=20000
log_freq=1000
csv_freq=100
checkpoint_dir="checkpoints_nstep"
log_dir="logs_nstep"
csv_path=os.path.join(log_dir,"training.csv")
episodios_path = os.path.join(log_dir, "episodios.csv")
os.makedirs(checkpoint_dir,exist_ok=True)
os.makedirs(log_dir,exist_ok=True)
device="cuda" if torch.cuda.is_available() else "cpu"

sched_eps={"inicio":0,"start":eps_start,"end":eps_end,"dur":decay_steps}
sched_lr={"inicio":lr_inicio,"start":learning_rate,"end":lr_end,"dur":lr_duracion}

def _rampa(sch,step):
    p=(step-sch["inicio"])/sch["dur"]
    p=min(max(p,0.0),1.0)
    return sch["start"]+p*(sch["end"]-sch["start"])

def eps_decae(step):
    return _rampa(sched_eps,step)
def lr_decae(step):                                                             # LR
    return _rampa(sched_lr,step)
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

def guardar_checkpoint(ruta, online, target, optimizer,step, episode, epsilon, best, ema=None):
    torch.save({
        "online_state":online.state_dict(),
        "target_state":target.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "step":step,
        "episode":episode,
        "epsilon":epsilon,
        "best":best,
        "ema_state":ema.state_dict() if ema is not None else None,
        "sched_eps":dict(sched_eps),
        "sched_lr":dict(sched_lr),}, ruta)

    
def cargar_checkpoint(ruta, online, target, optimizer, ema=None):
    checkpoint=torch.load(ruta,map_location=device,weights_only=False)
    online.load_state_dict(checkpoint["online_state"])
    target.load_state_dict(checkpoint["target_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if ema is not None:
        if checkpoint.get("ema_state") is not None:
            ema.modelo.load_state_dict(checkpoint["ema_state"])
            ema.n=10**9
        else:
            ema.modelo.load_state_dict(online.state_dict())
    if "sched_eps" in checkpoint:        # reanudar continúa las rampas que ya estaban en marcha
        sched_eps.update(checkpoint["sched_eps"])
        sched_lr.update(checkpoint["sched_lr"])
    return (checkpoint["step"],checkpoint["episode"],checkpoint["epsilon"],checkpoint["best"])

def escribir_csv(step, episode, epsilon, loss, reward, max_percent, buf_len, lr, ruta=csv_path):
    archivo_nuevo = not os.path.exists(ruta)

    with open(ruta,"a",newline="") as f:
        writer=csv.writer(f)
        if archivo_nuevo:
            writer.writerow(["step", "episode", "epsilon", "loss","reward_avg", "max_percent", "buffer_size", "lr"])
        writer.writerow([step, episode, f"{epsilon:.4f}", f"{loss:.4f}",f"{reward:.4f}", f"{max_percent:.2f}", buf_len, f"{lr:.2e}"])


def escribir_episodio(episode, step, pasos, percent_final, reward, percent_inicial, ruta=episodios_path):
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["episode", "step", "pasos", "percent_final", "reward", "percent_inicial"])
        w.writerow([episode, step, pasos, f"{percent_final:.2f}", f"{reward:.2f}", f"{percent_inicial:.3f}"])

#optimizer = optim.Adam(online.parameters(), lr=learning_rate)
def main():
    env=GD_env()
    #buf=replay_buffer(cap=capacity_buffer)
    #buf = replay_buffer_doble(cap=capacity_buffer, cap_muerte=cap_m, ventana_muerte=vm, frac_muerte=fm)
    buf = replay_buffer_tramos(ancho=5.0, cap_tramo=15000, exp_peso=0.5)
    online, target=crear_redes(device)
    optimizer=optim.Adam(online.parameters(),lr=learning_rate)
    nstep=NStepAcumulador(N_STEP,gamma)                                          # N-STEP
    ema=EMAPesos(online,decay_desde_ventana(ema_ventana,train_freq))             # EMA
    
    if resume_path and os.path.exists(resume_path):
        print(f"Pesos antiguos detectados, reanudando desde {resume_path}")
        step_inicial,episode,epsilon_inicial,best_max=\
            cargar_checkpoint(resume_path,online,target,optimizer,ema)
        if REINICIAR_EPS:
            sched_eps.update(inicio=step_inicial,start=eps_reinicio[0],end=eps_reinicio[1],dur=eps_reinicio[2])
            print(f"Epsilon reiniciado: {eps_reinicio[0]} -> {eps_reinicio[1]} en {eps_reinicio[2]} steps desde el step {step_inicial}")
        if REINICIAR_LR:
            sched_lr.update(inicio=step_inicial,start=lr_reinicio[0],end=lr_reinicio[1],dur=lr_reinicio[2])
            print(f"lr reiniciado: {lr_reinicio[0]:.1e} -> {lr_reinicio[1]:.1e} en {lr_reinicio[2]} steps desde el step {step_inicial}")
        print(f"Al reanudar: epsilon={eps_decae(step_inicial):.4f}, lr={lr_decae(step_inicial):.2e}, step={step_inicial}")
    else:
        step_inicial=1
        episode=0
        epsilon_inicial=eps_start
        best_max=0

    obs_frames,obs_states=env.reset()

    percent_inicial = obs_states[0] * 100
    episode_reward=0
    episode_steps=0
    episode_max_percent=0
    losses_recientes=deque(maxlen=100)
    rewards_recientes=deque(maxlen=100)

    percent_recientes = deque(maxlen=100)
    
    t_env = t_train = 0.0
    t_ini = time.time()

    gif=GrabadorGIF(carpeta="gifs")
    gif.activar_teclas()

    
    step=step_inicial
    try:
        for step in range(step_inicial,total_steps+1):
            epsilon=eps_decae(step)
            lr_actual=lr_decae(step)                                                 # LR
            for g in optimizer.param_groups:
                g["lr"]=lr_actual

            t0 = time.time()

            action=choose_accion(online,obs_frames,obs_states,epsilon,device)
            next_frames,next_state,reward,done,info=env.step(action)
            gif.agregar(next_frames[-1],action,step)

            t_env += time.time() - t0

            for t in nstep.agregar(obs_frames,obs_states,action,reward,next_frames,next_state,done):   # N-STEP
                buf.push(*t)
            episode_reward+=reward
            episode_steps+=1
            episode_max_percent=max(episode_max_percent,info["percent"])
            if step % train_freq==0 and len(buf)>=min_buffer:
                t1 = time.time()
                batch=buf.sample(batch_size)
                batch_gpu=prep_batch(batch,device)
                loss_valor=loss(online,target,batch_gpu,gamma**N_STEP,device)   # N-STEP
                paso_opt(optimizer,loss_valor,online)
                ema.actualizar(online)                                               # EMA
                losses_recientes.append(loss_valor.item())
                t_train += time.time() - t1
            if step%target_update_freq==0:
                update_target(online,target)
            if done:
            
                rewards_recientes.append(episode_reward)
                if episode_max_percent>best_max:
                    best_max=episode_max_percent
                    guardar_checkpoint(
                        os.path.join(checkpoint_dir, "best.pt"),
                        online, target, optimizer, step, episode,
                        epsilon, best_max, ema)
                    print(f"nuevo récord: {best_max:.1f}%")
                episode+=1
                escribir_episodio(episode, step, episode_steps, episode_max_percent, episode_reward, percent_inicial)
                episode_reward=0
                episode_steps=0
                percent_recientes.append(episode_max_percent)
                episode_max_percent=0
                obs_frames,obs_states=env.reset()
                percent_inicial = obs_states[0] * 100
            else:
                obs_frames,obs_states=next_frames,next_state
            if step%csv_freq==0:
                escribir_csv(step, episode, epsilon,
                             np.mean(losses_recientes) if losses_recientes else 0.0,
                             np.mean(rewards_recientes) if rewards_recientes else 0.0,
                             np.mean(percent_recientes) if percent_recientes else 0.0, len(buf), lr_actual)
            if step % save_freq == 0:
                ruta = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                guardar_checkpoint(ruta, online, target, optimizer,
                                   step, episode, epsilon, best_max, ema)
                ema.guardar(os.path.join(checkpoint_dir, f"ema_step_{step}.pt"))       # EMA
            if step % 1000 == 0:
                dt = time.time() - t_ini
                print(f"{1000/dt:.1f} steps/s | env {t_env:.1f}s | train {t_train:.1f}s de {dt:.1f}s")
                t_env = t_train = 0.0
                t_ini = time.time()
    except KeyboardInterrupt:
        print("Interrumpido: guardando antes de salir...")
        guardar_checkpoint(os.path.join(checkpoint_dir,"interrumpido.pt"),
                           online, target, optimizer, step, episode, epsilon, best_max, ema)
        ema.guardar(os.path.join(checkpoint_dir,"ema_interrumpido.pt"))
        return
    finally:
        env.close()

    print("-" * 60)
    print(f"Entrenamiento terminado. Best max%: {best_max:.1f}")
    guardar_checkpoint(
        os.path.join(checkpoint_dir, "final.pt"),
        online, target, optimizer, total_steps, episode,
        eps_decae(total_steps), best_max, ema)


if __name__ == "__main__":
    main()