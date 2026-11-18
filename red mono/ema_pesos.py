import copy
import torch


def decay_desde_ventana(ventana_steps, train_freq=1):
    """
    El EMA se actualiza una vez por cada paso del optimizador (cada `train_freq` steps del juego).
    La 'ventana' es aproximadamente cuántos steps del juego recuerda el promedio.
    """
    actualizaciones = max(1.0, ventana_steps / train_freq)
    return 1.0 - 1.0 / actualizaciones


class EMAPesos:
    def __init__(self, online, decay, calentamiento=True):
        # `online` ya debe tener sus capas Lazy inicializadas (crear_redes lo hace).
        self.modelo = copy.deepcopy(online).eval()
        for p in self.modelo.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.n = 0 if calentamiento else 10 ** 9

    @torch.no_grad()
    def actualizar(self, online):
        self.n += 1
        # al inicio promedia más rápido para no quedarse pegado a los pesos iniciales
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for pe, po in zip(self.modelo.parameters(), online.parameters()):
            pe.mul_(d).add_(po.detach(), alpha=1.0 - d)
        for be, bo in zip(self.modelo.buffers(), online.buffers()):
            be.copy_(bo)

    def state_dict(self):
        return self.modelo.state_dict()

    def guardar(self, ruta):
        torch.save({"online_state": self.modelo.state_dict()}, ruta)

    def cargar(self, ruta, device=None):
        """Parte de un promedio ya hecho (por ejemplo promedio_820_900.pt) sin calentamiento."""
        ck = torch.load(ruta, map_location=device, weights_only=False)
        self.modelo.load_state_dict(ck["online_state"])
        self.n = 10 ** 9