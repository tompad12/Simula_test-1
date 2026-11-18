import time
from gd_lector_estado import GDStateReader

r = GDStateReader()
s = r.read()
ult_percent, t_cambio = s["percent"], time.time()
ult_attempts = s["attempts"]
congelado, t_congelado = False, 0.0

while True:
    s = r.read()
    ahora = time.time()

    if s["attempts"] != ult_attempts:
        ult_attempts = s["attempts"]
        if congelado:
            print(f"attempts subió {ahora - t_congelado:.2f} s después de que percent se congeló")
        else:
            print("attempts subió sin congelamiento previo de percent")

    if s["percent"] != ult_percent:
        ult_percent, t_cambio, congelado = s["percent"], ahora, False
    elif not congelado and s["percent"] > 0 and ahora - t_cambio > 0.08:
        congelado, t_congelado = True, t_cambio
        print(f"percent congelado en {s['percent']:.2f}% (posible choque)")

    time.sleep(0.005)