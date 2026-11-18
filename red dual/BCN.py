import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from GD_DQN import GD_DQN

device = "cuda" if torch.cuda.is_available() else "cpu"


class GDDemo(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.frames = data["frames"]
        self.states = data["states"]
        self.actions = data["actions"]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return self.frames[i], self.states[i], self.actions[i]


def bc_loss(online, batch, device):
    frames, states, actions = batch
    frames = frames.float().to(device) / 255.0
    states = states.float().to(device)
    actions = actions.long().to(device)

    q_values = online(frames, states)            # (B, 2)

    # Target: 1.0 en la acción tomada, 0.0 en la otra
    target = torch.zeros_like(q_values)
    target.scatter_(1, actions.unsqueeze(1), 1.0)

    return nn.MSELoss()(q_values, target)

def evaluar(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for frames, states, actions in loader:
            actions_gpu = actions.long().to(device)
            loss = bc_loss(model, (frames, states, actions), device)
            total_loss += loss.item()

            frames = frames.float().to(device) / 255.0
            states = states.float().to(device)
            pred = model(frames, states).argmax(dim=1)
            total_correct += (pred == actions_gpu).sum().item()
            total_samples += len(actions)

    model.train()
    return total_loss / len(loader), total_correct / total_samples


def train_bc():
    dataset = GDDemo("demos/human_play_01.npz")

    # Diagnóstico
    print("=" * 60)
    acciones = dataset.actions
    dist = np.bincount(acciones)
    print(f"Total transiciones: {len(acciones)}")
    print(f"Distribución: {dist}")
    print(f"Idle: {(acciones == 0).mean() * 100:.1f}%")
    print(f"Hold: {(acciones == 1).mean() * 100:.1f}%")

    if dist[0] == 0 or (len(dist) > 1 and dist[1] == 0):
        print("❌ Todas las acciones son iguales. Revisa record.py.")
        return

    # Split
    n = len(dataset)
    n_train = int(n * 0.9)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    print(f"Train: {n_train} | Val: {n_val}")

    # Modelo
    model = GD_DQN().to(device)
    dummy_x = torch.zeros(1, 4, 84, 84, device=device)
    dummy_s = torch.zeros(1, 4, device=device)
    model(dummy_x, dummy_s)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("=" * 60)
    print("ENTRENAMIENTO")
    print("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(20):
        # Train
        model.train()
        train_loss = 0.0
        for frames, states, actions in train_loader:
            loss = bc_loss(model, (frames, states, actions), device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        val_loss, val_acc = evaluar(model, val_loader, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " ⭐"
            torch.save({"online_state": model.state_dict()},
                       "checkpoints/bc_pretrained.pt")

        print(f"epoch {epoch:2d}: "
              f"train={train_loss:.6f}  "
              f"val={val_loss:.6f}  "
              f"val_acc={val_acc*100:.1f}%{marker}")

    print(f"\nMejor val_loss: {best_val_loss:.6f}")
    print("Guardado en checkpoints/bc_pretrained.pt")


if __name__ == "__main__":
    train_bc()