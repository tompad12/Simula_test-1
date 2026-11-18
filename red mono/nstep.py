from collections import deque


class NStepAcumulador:
    def __init__(self, n, gamma):
        assert n >= 1
        self.n = n
        self.gamma = gamma
        self.cola = deque()

    def _armar(self, siguiente_frames, siguiente_estado, done):
        f0, s0, a0, _ = self.cola[0]
        retorno = 0.0
        for k, (_, _, _, r) in enumerate(self.cola):
            retorno += (self.gamma ** k) * r
        return (f0, s0, a0, retorno, siguiente_frames, siguiente_estado, done)

    def agregar(self, frames, estado, accion, recompensa, sig_frames, sig_estado, done):
        """Devuelve la lista de transiciones listas para el buffer (0, 1 o varias)."""
        self.cola.append((frames, estado, accion, recompensa))
        salida = []
        if done:
            # El episodio terminó: cada transición pendiente cierra en el estado terminal
            # (done=True, sin bootstrap), con su propio horizonte más corto.
            while self.cola:
                salida.append(self._armar(sig_frames, sig_estado, True))
                self.cola.popleft()
        elif len(self.cola) == self.n:
            salida.append(self._armar(sig_frames, sig_estado, False))
            self.cola.popleft()
        return salida

    def reiniciar(self):
        self.cola.clear()