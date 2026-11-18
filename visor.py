# visor_gd.py
"""
Visor en tiempo real de las capturas de Geometry Dash.

- Reutiliza GeometryDashCapture de capturar_juego.py
- Reutiliza GDStateReader de gd_lector_estado.py (opcional)
- Muestra en una ventana cv2 la imagen procesada (84x84) ampliada
  y, opcionalmente, el ROI original.

Controles:
    Q / ESC  -> Salir
    R        -> Alternar vista: procesado <-> raw
    G        -> (re)centrar la ventana sobre Geometry Dash
    S        -> Guardar un frame en disco
"""

import time
import cv2
import numpy as np

from capturar_juego import GeometryDashCapture, geometry_dash_activo, IMAGE_SIZE

# El lector de estado es opcional: si no existe el mod de Geode, seguimos funcionando.
try:
    from gd_lector_estado import GDStateReader
    _TIENE_LECTOR = True
except Exception as e:
    print(f"[aviso] No se pudo importar GDStateReader: {e}")
    _TIENE_LECTOR = False


# ---------- Utilidades de dibujo ----------

def _escala_para_mostrar(img, tamaño_min=420):
    """Amplía una imagen pequeña (84x84) a un tamaño cómodo con vecino más cercano."""
    h, w = img.shape[:2]
    if h >= tamaño_min:
        return img
    factor = max(1, tamaño_min // max(h, w))
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def _superponer_hud(img_bgr, fps, estado=None, modo="procesado"):
    """Dibuja un pequeño HUD con info de FPS, modo y estado del juego."""
    lineas = [f"Modo: {modo}", f"FPS: {fps:5.1f}"]
    if estado is not None:
        lineas.append(
            f"Attempts: {estado['attempts']}  "
            f"Percent: {estado['percent']:.1f}%  "
            f"Vehicle: {estado['vehicle']}  "
            f"UpsideDown: {estado['is_upside_down']}"
        )
    lineas.append("Q/ESC: salir   R: alternar   G: recentrar   S: guardar")

    y = 22
    for linea in lineas:
        cv2.putText(img_bgr, linea, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_bgr, linea, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22
    return img_bgr


# ---------- Programa principal ----------

def main():
    print("Iniciando visor de Geometry Dash...")
    print("Asegúrate de tener Geometry Dash abierto y en primer plano.")

    # 1) Capturador
    capturador = GeometryDashCapture(image_size=IMAGE_SIZE)

    # 2) Lector de estado (si está disponible)
    lector = None
    if _TIENE_LECTOR:
        try:
            lector = GDStateReader()
            print("[ok] Lector de estado conectado a la memoria compartida.")
        except Exception as e:
            print(f"[aviso] No se pudo conectar al lector de estado: {e}")
            lector = None

    ventana = "Geometry Dash - Visor"
    cv2.namedWindow(ventana, cv2.WINDOW_AUTOSIZE)

    modo = "procesado"          # "procesado" o "raw"
    guardado_idx = 0

    # Para medir FPS
    t_prev = time.time()
    fps = 0.0

    try:
        while True:
            t_ahora = time.time()
            dt = t_ahora - t_prev
            t_prev = t_ahora
            if dt > 0:
                # Suavizado exponencial para que no parpadee el número
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # --- Captura ---
            try:
                raw = capturador.capture_raw()          # BGRA ROI
                procesado = capturador.process(raw)     # 84x84 gris
            except Exception as e:
                # Si el juego se cierra o la ventana desaparece, esperamos un poco
                print(f"[aviso] Falló la captura: {e}")
                time.sleep(0.5)
                continue

            # --- Lectura de estado ---
            estado = None
            if lector is not None:
                try:
                    estado = lector.read()
                except Exception as e:
                    print(f"[aviso] Falló lectura de estado: {e}")
                    lector = None

            # --- Preparar imagen para mostrar ---
            if modo == "procesado":
                vis = _escala_para_mostrar(procesado)
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            else:
                # raw viene en BGRA; lo pasamos a BGR para mostrarlo
                vis_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

            # Aviso si GD no está en primer plano (no detenemos, solo avisamos)
            activo = geometry_dash_activo()
            if not activo:
                cv2.putText(vis_bgr, "GD NO ESTA EN PRIMER PLANO",
                            (10, vis_bgr.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            vis_bgr = _superponer_hud(vis_bgr, fps, estado, modo)
            cv2.imshow(ventana, vis_bgr)

            # --- Entrada de teclado ---
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord('q'), 27):     # q o ESC
                break
            elif tecla == ord('r'):
                modo = "raw" if modo == "procesado" else "procesado"
                print(f"[info] Modo cambiado a: {modo}")
            elif tecla == ord('g'):
                # Recentra el ROI por si movieron la ventana del juego
                capturador.actualizar_geometria()
                print("[info] Geometría del ROI actualizada.")
            elif tecla == ord('s'):
                nombre = f"captura_{guardado_idx:04d}.png"
                if modo == "procesado":
                    cv2.imwrite(nombre, procesado)
                else:
                    cv2.imwrite(nombre, cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR))
                guardado_idx += 1
                print(f"[info] Guardado {nombre}")

    finally:
        # Limpieza
        cv2.destroyAllWindows()
        try:
            capturador.close()
        except Exception:
            pass
        if lector is not None:
            try:
                lector.close()
            except Exception:
                pass
        print("Visor cerrado.")


if __name__ == "__main__":
    main()