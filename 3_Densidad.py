# Análisis de densidad

import librosa
import sys
import numpy as np
import math

# Parámetros de configuración
FRAMES_POR_SEGUNDO = 22050.0
SEGMENTO_SEGUNDOS = 10
SEGMENTO_FRAMES = SEGMENTO_SEGUNDOS * FRAMES_POR_SEGUNDO

# Función para calcular los bloques de densidad de un archivo de audio
def calcular_bloques(filename):
    y, sr = librosa.load(filename, sr=None)
    duracion = len(y) / FRAMES_POR_SEGUNDO
    cantidad_bloques = int(math.ceil(duracion / SEGMENTO_SEGUNDOS))
    print(f"Duración del archivo {filename}: {duracion} segundos")
    print(f"Cantidad de bloques: {cantidad_bloques}")

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='samples') / FRAMES_POR_SEGUNDO
    bloques = np.zeros(cantidad_bloques)

    for onset in onset_frames:
        indice_onset = int(onset / SEGMENTO_SEGUNDOS)
        if indice_onset < cantidad_bloques:
            bloques[indice_onset] += 1

    return bloques

# Archivos de audio a comparar
archivo1 = sys.argv[1]
archivo2 = sys.argv[2]

# Obtener los bloques de densidad para ambos archivos
bloques1 = calcular_bloques(archivo1)
bloques2 = calcular_bloques(archivo2)

# Ajustar los tamaños de los bloques para que coincidan
if len(bloques1) < len(bloques2):
    bloques1 = np.append(bloques1, np.zeros(len(bloques2) - len(bloques1)))
elif len(bloques2) < len(bloques1):
    bloques2 = np.append(bloques2, np.zeros(len(bloques1) - len(bloques2)))

# Calcular la diferencia entre las densidades de ambos archivos
diferencia = np.sum(np.abs(bloques1 - bloques2))
print(f"La diferencia entre las densidades sonoras de los archivos es de {diferencia}")
