from collections import deque
import random
import numpy as np


class replay_buffer_tramos:
    """Un buffer por tramo de progreso (percent).
    El batch se reparte entre tramos con peso ~ tamaño ** exp_peso:
    los tramos con pocos datos (el frontera) pesan más que en un buffer normal,
    pero sin repetir hasta el cansancio un tramo casi vacío."""

    def __init__(self, ancho=5.0, cap_tramo=15000, exp_peso=0.5):
        self.ancho = ancho            # ancho de cada tramo, en % de avance
        self.cap_tramo = cap_tramo    # transiciones máximas por tramo
        self.exp_peso = exp_peso      # 1 = proporcional al tamaño, 0 = igual para cada tramo
        self.tramos = {}              # índice de tramo -> deque de transiciones

    def __len__(self):
        return sum(len(d) for d in self.tramos.values())

    def push(self, frames, state, action, reward, next_frames, next_state, done):
        i = int(state[0] * 100 / self.ancho)      # state[0] = percent / 100
        if i not in self.tramos:
            self.tramos[i] = deque(maxlen=self.cap_tramo)
        self.tramos[i].append((frames, state, action, reward, next_frames, next_state, done))

    def sample(self, batch_size):
        idx = [i for i, d in self.tramos.items() if len(d) > 0]
        pesos = np.array([len(self.tramos[i]) ** self.exp_peso for i in idx])
        cuentas = np.random.multinomial(batch_size, pesos / pesos.sum())

        batch = []
        for i, c in zip(idx, cuentas):
            if c > 0:
                batch += random.choices(self.tramos[i], k=int(c))   # con reemplazo

        frames, states, actions, rewards, next_frames, next_states, dones = zip(*batch)
        return (
            np.stack(frames, axis=0).astype(np.uint8),
            np.stack(states, axis=0).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_frames, axis=0).astype(np.uint8),
            np.stack(next_states, axis=0).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )