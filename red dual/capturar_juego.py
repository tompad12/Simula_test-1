
# Paquete para tomar screenshots eficientemente
import mss
# Paquete para procesado de imagenes (escala de grisses y reescalamiento)
import cv2
# Paquete de utilidad para detectar ventanas y su ubicación
import win32gui

import numpy as np
import keyboard


 # Verificamos que el Geometry Dash sea pestaña activa (no en 2do plano, solo para tests)
def geometry_dash_activo():
    hwnd = win32gui.GetForegroundWindow()
    hwnd_gd = win32gui.FindWindow(None, "Geometry Dash")
    return (hwnd == hwnd_gd and not win32gui.IsIconic(hwnd)) 

# Basicamente encuentra donde la zona donde está el GD en pantalla para luego tomarle pantallazo
def obtener_region_gd():
    gd_abierto = win32gui.FindWindow(None, "Geometry Dash")
    if not gd_abierto:
        raise RuntimeError("Geometry Dash no está abierto.")

    # recuperamos la ubicacion coordenada del gd
    left, top, right, bot = win32gui.GetClientRect(gd_abierto)
    
    # coordenadas absolutas
    screen_left, screen_top = win32gui.ClientToScreen(gd_abierto, (left, top))
    
    ancho = right - left
    alto = bot - top

    return {"left": screen_left, "top": screen_top, "width": ancho, "height": alto}

# Dimension de la imagen de salida, modificar si lo ven necesario
IMAGE_SIZE = (84, 84)

class GeometryDashCapture:

    def __init__(self, image_size=IMAGE_SIZE):
        self.image_size = image_size
        self.sct = mss.mss()
        
        # Calculamos la zona y zoom a aplicar. Ojo que solo lo llamamos UNA vez al iniciar la clase.
        self.actualizar_geometria()

    def actualizar_geometria(self):
        self.region = obtener_region_gd()
        w = self.region["width"]
        h = self.region["height"]

        # roi/zoom dinámico: (y_min, y_max, x_min, x_max)
        # con esos parametros justo enfoca antes del icono hasta los obstáculos siguientes
        self.roi = (0, h, int(w * 0.20), int(w * 0.85))

    def capture_raw(self):
        """Captura únicamente el ROI elegido."""
        frame = np.asarray(self.sct.grab(self.region))
        y_min, y_max, x_min, x_max = self.roi
        return frame[y_min:y_max, x_min:x_max]

    # Se reescala y pasa a grises basicamente
    def process(self, frame):
        """Convierte BGRA -> Grayscale -> 84x84."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return cv2.resize(frame, self.image_size, interpolation=cv2.INTER_AREA)

    def capture(self):
        raw = self.capture_raw()
        return self.process(raw)

    def close(self):
        self.sct.close()

# Con esta funcion pueden probar la captura que se realiza
def modo_captura_manual(capture, tecla_c="c",tecla_p="p"):
    """
    c   -> captura
    P   -> pausa/reanuda
    
    Imprime imagen de:
        ROI original
        ROI procesado 84x84
    """
    print("Modo captura manual")
    print("C   -> Capturar")
    print("P   -> Salir")

    while True:
        # Tecla para pausar
        if keyboard.is_pressed(tecla_p):
            print("Saliendo...")
            break

        # Tecla para capturar el gd
        if keyboard.is_pressed(tecla_c):
            # Si está en 2do plano se avisa
            if geometry_dash_activo():
                raw = capture.capture_raw()
                processed = capture.process(raw)
                raw_rgb = cv2.cvtColor(raw,cv2.COLOR_BGRA2RGB)

                # Se grafica
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1,2,figsize=(12, 5))
                axes[0].imshow(raw_rgb)
                axes[0].set_title(f"Original ({raw.shape[1]}×{raw.shape[0]})")
                axes[0].axis("off")

                axes[1].imshow(processed,cmap="gray",interpolation="nearest")
                axes[1].set_title(
                    f"Entrada DQN ({processed.shape[1]}×{processed.shape[0]})")
                axes[1].axis("off")
                plt.show()

            else:
                print("Geometry Dash no está activo.")