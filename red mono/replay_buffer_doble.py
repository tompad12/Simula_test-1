from collections import deque
import random
import numpy as np
from replay_buffer import replay_buffer


class replay_buffer_doble(replay_buffer):
    def __init__(self, cap=50000, cap_muerte=10000,
                 ventana_muerte=5, frac_muerte=0.125):
        super().__init__(cap=cap)
        self.buffer_muerte = deque(maxlen=cap_muerte)
        self.reciente = deque(maxlen=ventana_muerte)  # últimos steps del episodio actual
        self.frac_muerte = frac_muerte

    def push(self, frames, state, action, reward, next_frames, next_state, done):
        trans = (frames, state, action, reward, next_frames, next_state, done)
        self.buffer.append(trans)      # buffer uniforme, igual que antes
        self.reciente.append(trans)    # ventana móvil del episodio actual

        if done:
            murio = (reward < 0)         # muerte = reward negativo; victoria = positivo
            if murio:
                self.buffer_muerte.extend(self.reciente)
            self.reciente.clear()      # el próximo episodio empieza limpio

    def sample(self, batch_size):
        n_muerte = min(int(batch_size * self.frac_muerte), len(self.buffer_muerte))
        n_uniforme = batch_size - n_muerte

        batch = random.sample(self.buffer, n_uniforme)
        if n_muerte > 0:
            batch += random.sample(self.buffer_muerte, n_muerte)

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