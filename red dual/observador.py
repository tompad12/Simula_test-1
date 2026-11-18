# gd_observador.py
"""
Extrae, en una sola llamada, la observación (imagen) y el estado del juego.

Combina:
    GeometryDashCapture   (capturar_juego.py)   -> imagen 84x84 uint8
    GDStateReader         (gd_lector_estado.py) -> dict con percent, attempts, ...

No guarda nada en disco ni entrena nada. Solo devuelve los datos.
"""

import time
import numpy as np

from capturar_juego import GeometryDashCapture, geometry_dash_activo, IMAGE_SIZE
from gd_lector_estado import GDStateReader


class GDObservador:
    """
    Interfaz mínima para leer la observación actual del juego.

    Uso típico:

        obs = GDObservador()
        while True:
            frame, estado = obs.leer()
            # ... tu red neuronal aquí ...
        obs.cerrar()

    Detecta automáticamente el reinicio de un intento: si `attempts`
    cambia entre dos lecturas, devuelve `episodio_nuevo=True`.
    """

    def __init__(self, image_size=IMAGE_SIZE, comprobar_primer_plano=False):
        self.captura = GeometryDashCapture(image_size=image_size)
        self.lector = GDStateReader()
        self.comprobar_primer_plano = comprobar_primer_plano
        self._attempts_previo = None
        self._t_previo = None

    # ---------------- API pública ----------------

    def leer(self):
        """
        Devuelve (frame, estado, info).

        frame  : np.ndarray (84, 84) uint8  -> entrada de la CNN
        estado : dict  {"percent": float, "attempts": int,
                        "vehicle": int, "is_upside_down": bool}
        info   : dict  {"episodio_nuevo": bool,
                        "dt": float,             # segundos desde la lectura anterior
                        "gd_activo": bool}       # solo se rellena si comprobar_primer_plano=True
        """
        # 1) Leer estado ANTES (para detectar resets que ocurran durante la captura)
        estado_a = self.lector.read()

        # 2) Capturar la imagen
        frame = self.captura.capture()

        # 3) Leer estado DESPUÉS
        estado_b = self.lector.read()

        # Si los attempts cambiaron entre las dos lecturas, hubo reset
        hubo_reset = estado_a["attempts"] != estado_b["attempts"]

        # El estado que devolvemos es el "de después" porque corresponde
        # al frame que acabamos de capturar (mss y la memoria son casi simultáneos).
        estado = estado_b

        # 4) Detectar inicio de episodio
        episodio_nuevo = False
        if self._attempts_previo is None or self._attempts_previo != estado["attempts"]:
            episodio_nuevo = True
        self._attempts_previo = estado["attempts"]

        # 5) dt
        t_ahora = time.time()
        dt = 0.0 if self._t_previo is None else (t_ahora - self._t_previo)
        self._t_previo = t_ahora

        info = {
            "episodio_nuevo": episodio_nuevo or hubo_reset,
            "dt": dt,
            "gd_activo": True,
        }

        if self.comprobar_primer_plano:
            info["gd_activo"] = geometry_dash_activo()

        return frame, estado, info

    def reintentar_geometria(self):
        """Recalcula el ROI (útil si moviste la ventana del juego)."""
        self.captura.actualizar_geometria()

    def cerrar(self):
        try:
            self.captura.close()
        except Exception:
            pass
        try:
            self.lector.close()
        except Exception:
            pass

    # Context manager para no olvidar cerrar
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cerrar()


# ---------------- Demo mínimo ----------------

if __name__ == "__main__":
    with GDObservador(comprobar_primer_plano=True) as obs:
        print("Leyendo observaciones. Ctrl+C para salir.")
        try:
            while True:
                frame, estado, info = obs.leer()

                # Aquí tú harías:  accion = mi_red(frame, estado)
                # y luego enviarías la acción al juego con keyboard.press(...)

                print(
                    f"frame={frame.shape} dtype={frame.dtype} | "
                    f"percent={estado['percent']:6.1f}% "
                    f"attempts={estado['attempts']:4d} "
                    f"vehicle={estado['vehicle']} "
                    f"upside_down={estado['is_upside_down']} | "
                    f"nuevo={info['episodio_nuevo']} "
                    f"dt={info['dt']*1000:5.1f}ms"
                )
                time.sleep(1 / 30)   # ~30 Hz, ajusta a lo que necesites

        except KeyboardInterrupt:
            print("Saliendo.")