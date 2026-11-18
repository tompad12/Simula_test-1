import os
import time
import threading
import numpy as np
from PIL import Image, ImageDraw

# Pillow >= 9.1 guarda los filtros en Image.Resampling; las versiones viejas los tienen en Image.
NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")


class GrabadorGIF:
    def __init__(self, carpeta="gifs", escala=4, marcar_accion=True, max_pausa=0.1):
        self.carpeta = carpeta
        self.escala = escala
        self.marcar_accion = marcar_accion
        self.max_pausa = max_pausa          # un hueco largo (p. ej. un reset) se acorta a esto, en segundos
        self.activo = False
        self._pedido = None                 # lo escribe el hilo del teclado; lo lee el bucle principal
        self._imgs = []
        self._tiempos = []
        self._t_fin = 0.0
        self._step = 0
        os.makedirs(carpeta, exist_ok=True)

    # ---- teclas
    def pedir(self, segundos):
        """Pide una grabación; empieza en el siguiente step. Seguro de llamar desde otro hilo."""
        if not self.activo:
            self._pedido = float(segundos)

    def activar_teclas(self, tecla_5s="5", tecla_10s="0"):
        try:
            import keyboard
            keyboard.add_hotkey(tecla_5s, lambda: self.pedir(5))
            keyboard.add_hotkey(tecla_10s, lambda: self.pedir(10))
            print(f"GIF: {tecla_5s.upper()} graba 5 s, {tecla_10s.upper()} graba 10 s")
        except Exception as e:
            print(f"GIF: no se pudieron registrar las teclas ({e}). Usa gif.pedir(segundos).")

    # ---- llamar una vez por step
    def agregar(self, frame, accion=0, step=0):
        if not self.activo and self._pedido is not None:
            self.activo = True
            self._t_fin = time.time() + self._pedido
            self._step = step
            self._imgs, self._tiempos = [], []
            print(f"GIF: grabando {self._pedido:.0f} s...")
            self._pedido = None
        if not self.activo:
            return
        ahora = time.time()
        img = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8))          # 84x84 -> modo 'L'
        img = img.resize((frame.shape[1] * self.escala, frame.shape[0] * self.escala), NEAREST)
        if self.marcar_accion and accion == 1:
            d = ImageDraw.Draw(img)
            d.rectangle([3, 3, 3 + 3 * self.escala, 3 + 3 * self.escala], fill=255, outline=0)
        self._imgs.append(img)
        self._tiempos.append(ahora)
        if ahora >= self._t_fin:
            imgs, tiempos, paso = self._imgs, self._tiempos, self._step
            self.activo = False
            self._imgs, self._tiempos = [], []
            ruta = os.path.join(self.carpeta, f"gif_{time.strftime('%Y%m%d_%H%M%S')}_step{paso}.gif")
            threading.Thread(target=self._guardar, args=(imgs, tiempos, ruta), daemon=True).start()

    # ---- guardado (en un hilo aparte)
    def _duraciones_ms(self, tiempos):
        """Duración de cada frame en ms. Los GIF miden el tiempo en centésimas, así que se reparte el
        redondeo entre frames para que el GIF dure lo mismo que la grabación real."""
        dts = np.diff(tiempos)
        dts = np.clip(dts, 0.02, self.max_pausa)
        dts = np.append(dts, np.median(dts) if len(dts) else 0.03)
        acum = np.cumsum(dts) * 100.0                      # centésimas acumuladas
        objetivo = np.maximum(2, np.round(acum)).astype(int)
        durs, previo = [], 0
        for o in objetivo:
            cs = max(2, o - previo)                        # los navegadores no respetan menos de 2 cs
            durs.append(cs * 10)
            previo += cs
        return durs

    def _guardar(self, imgs, tiempos, ruta):
        try:
            durs = self._duraciones_ms(tiempos)
            imgs[0].save(ruta, save_all=True, append_images=imgs[1:], duration=durs, loop=0)
            print(f"GIF guardado: {ruta} ({len(imgs)} frames, {sum(durs)/1000:.1f} s)")
        except Exception as e:
            print(f"GIF: error al guardar: {e}")