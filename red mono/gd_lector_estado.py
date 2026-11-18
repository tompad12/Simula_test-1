# Permite a Python acceder a un bloque de memoria RAM
import mmap
# Util para leer los bytes crudos (0's y 1's) de la memoria
import struct

# Nombre de la memoria compartida (con el mod de Geode) creada en C++
SHM_NAME = "Local\\GDRLSharedMem"

# Formato de lectura de los datos: float, int, int, bool ("fii?")
FMT = "<fii?"
# Tamaño total de los datos (bytes)
SIZE = struct.calcsize(FMT)  

# Clase para leer la información otorgada por el mod de Geode en GD
class GDStateReader:
    def __init__(self):
        # Conecta a la memoria compartida en la RAM
        self._shm = mmap.mmap(-1, SIZE, tagname=SHM_NAME)

    def read(self) -> dict:
        # Vuelve al inicio del bloque de memoria
        self._shm.seek(0)
        
        # Lee los bytes de información
        data = self._shm.read(SIZE)
        percent, attempts, vehicle, is_upside_down = struct.unpack(FMT, data)
        
        # Retornamos como diccionario los datos
        return {"percent": percent,"attempts": attempts, 
                "vehicle": vehicle,"is_upside_down": is_upside_down}

    def close(self):
        # Cierra la conexión
        self._shm.close()