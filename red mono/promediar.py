import sys
import torch

def promediar(rutas, salida):
    estados = []
    for r in rutas:
        ck = torch.load(r, map_location="cpu", weights_only=False)
        estados.append(ck["online_state"])
    promedio = {k: sum(e[k].float() for e in estados) / len(estados) for k in estados[0]}
    ck["online_state"] = promedio        # reutiliza el último checkpoint como envoltorio
    torch.save(ck, salida)
    print(f"Promedio de {len(rutas)} checkpoints guardado en {salida}")

if __name__ == "__main__":
    promediar(sys.argv[2:], sys.argv[1])