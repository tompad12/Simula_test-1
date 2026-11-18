"""
Revisa una demo grabada con record.py y, opcionalmente, anula una columna del vector de estado.
 
Uso:
    python revisar_demo.py demos/human_play_02.npz
    python revisar_demo.py demos/human_play_02.npz --anular 1      # crea demos/human_play_02_sin1.npz
 
Solo necesita numpy.
"""
import sys
import numpy as np
 
 
def main():
    ruta = sys.argv[1]
    d = np.load(ruta)
    frames, states, actions = d["frames"], d["states"], d["actions"]
    attempts = d["attempts"] if "attempts" in d.files else None
    n = len(actions)
 
    print(f"Transiciones: {n}  (~{n/37/60:.1f} min a 37 Hz)")
    print(f"frames {frames.shape}  states {states.shape}")
 
    # 1) Columnas del vector de estado: una columna que valga decenas o cientos delata a 'attempts'
    print("\nVector de estado (por columna):")
    print("  col |      min |      max |     media |      std")
    for j in range(states.shape[1]):
        c = states[:, j].astype(np.float64)
        print(f"  {j:3d} | {c.min():8.3f} | {c.max():8.3f} | {c.mean():9.3f} | {c.std():8.3f}")
 
    # 2) Acciones: cuántos saltos hay y cuánto dura cada pulsación
    onsets = np.flatnonzero((actions[1:] == 1) & (actions[:-1] == 0)) + 1
    print(f"\nHold: {100*actions.mean():.1f}%   pulsaciones (0->1): {len(onsets)}")
    if len(onsets):
        largos, run = [], 0
        for a in actions:
            if a == 1:
                run += 1
            elif run:
                largos.append(run)
                run = 0
        if run:
            largos.append(run)
        largos = np.array(largos)
        print(f"Duración de cada pulsación (steps): mediana {np.median(largos):.0f}, "
              f"p90 {np.percentile(largos, 90):.0f}, máx {largos.max()}")
        print("  (a 37 Hz, 7 steps ~ 0.19 s; una pulsación muy larga es normal en la nave, rara en el cubo)")
 
    # 3) Intentos: cuántos hay y hasta dónde llega cada uno
    if attempts is not None:
        cortes = np.flatnonzero(np.diff(attempts) != 0) + 1
        ini = np.r_[0, cortes]
        fin = np.r_[cortes, n]
        print(f"\nIntentos distintos en la grabación: {len(ini)}  (attempts de {attempts.min()} a {attempts.max()})")
        print("Si 'attempts' entra crudo al vector de estado, la red ve ese número como una entrada más:")
        print("durante la evaluación toma valores que nunca vio en la demo.")
 
    # 4) Anular una columna
    if "--anular" in sys.argv:
        col = int(sys.argv[sys.argv.index("--anular") + 1])
        nuevos = states.copy()
        nuevos[:, col] = 0.0
        salida = ruta.replace(".npz", f"_sin{col}.npz")
        datos = {k: d[k] for k in d.files}
        datos["states"] = nuevos
        np.savez_compressed(salida, **datos)
        print(f"\nColumna {col} puesta a 0. Guardado en {salida}")
 
 
if __name__ == "__main__":
    main()