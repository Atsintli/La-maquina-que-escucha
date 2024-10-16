# Script Maestro de Análisis de Audio

import os
import sys
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import csv
import pandas as pd
import tensorflow as tf
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Parámetros de configuración
NUM_MAX_CLASES = 8
FACTOR_DE_REDUCCION = 256
SEGMENTO_SEGUNDOS = 10
FRAMES_POR_SEGUNDO = 22050.0
k = 8
max_iterations = 100

# Definir funciones comunes
def crear_directorio(nombre_archivo, sufijo=""):
    nombre_directorio = os.path.splitext(os.path.basename(nombre_archivo))[0] + sufijo
    if not os.path.exists(nombre_directorio):
        os.makedirs(nombre_directorio)
    print(f"Directorio creado: {nombre_directorio}")
    return nombre_directorio

def cargar_audio(nombre_archivo):
    print(f"Cargando archivo de audio: {nombre_archivo}")
    y, sr = librosa.load(nombre_archivo)
    print(f"Audio cargado. Tasa de muestreo: {sr}, Número de muestras: {len(y)}")
    return y, sr

# Segmentación de audio
def segmentar_audio(nombre_archivo):
    nombre_directorio = crear_directorio(nombre_archivo)
    y, sr = cargar_audio(nombre_archivo)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    picos = librosa.util.peak_pick(onset_env, pre_max=3, post_max=10, pre_avg=7, post_avg=7, delta=0.20, wait=0.01)
    tiempos = librosa.frames_to_samples(picos)
    print(f"Número de segmentos = {len(tiempos)}")

    # Guardar segmentos
    print("Guardando segmentos de audio...")
    for idx in range(len(tiempos) - 1):
        inicio, fin = tiempos[idx], tiempos[idx + 1]
        nombre_segmento = os.path.join(nombre_directorio, f"{nombre_directorio}_{idx:05d}.wav")
        sf.write(nombre_segmento, y[inicio:fin], sr)
        print(f"Segmento {idx} guardado: {inicio} a {fin}")
    # Último segmento
    inicio = tiempos[-1]
    nombre_segmento = os.path.join(nombre_directorio, f"{nombre_directorio}_{len(tiempos) - 1:05d}.wav")
    sf.write(nombre_segmento, y[inicio:], sr)
    print(f"Último segmento guardado: {inicio} a {len(y)}")

    return nombre_directorio

# Extracción de características y almacenamiento en CSV
def extraer_caracteristicas(carpeta_segmentos):
    carpeta_salida = f"{carpeta_segmentos}_mfccs"
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    archivo_csv = os.path.join(carpeta_salida, 'caracteristicas_mfcc.csv')
    archivos_audio = glob.glob(os.path.join(carpeta_segmentos, '*.wav'))

    # Escribir encabezado y datos en el archivo CSV
    with open(archivo_csv, mode='w', newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(['Archivo'] + [f'MFCC{i+1}' for i in range(12)])
        for archivo_audio in archivos_audio:
            nombre_base = os.path.splitext(os.path.basename(archivo_audio))[0]
            y, sr = librosa.load(archivo_audio)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12).T, axis=0)
            writer.writerow([nombre_base] + mfccs.tolist())
            print(f"Características extraídas y guardadas para: {nombre_base}")
    return archivo_csv

# Clasificación de los segmentos
def clasificar_segmentos(csv_path):
    data = pd.read_csv(csv_path)
    nombres = data['Archivo'].tolist()
    X = data.drop(columns=['Archivo']).to_numpy()

    # Normalizar los datos para mejorar la precisión
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reducción de dimensionalidad con PCA para mejorar la clasificación
    pca = PCA(n_components=min(8, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    centroides = X_pca[:k, :]
    i, convergencia = 0, False

    with tf.compat.v1.Session() as sesion:
        sesion.run(tf.compat.v1.local_variables_initializer())
        while not convergencia and i < max_iterations:
            i += 1
            vectores_expandidos = tf.expand_dims(X_pca, 0)
            centroides_expandidos = tf.expand_dims(centroides, 1)
            distancias = tf.reduce_sum(tf.square(tf.subtract(vectores_expandidos, centroides_expandidos)), 2)
            Y = tf.argmin(distancias, 0)
            Y_val = sesion.run(Y)
            centroides = sesion.run(tf.math.unsorted_segment_sum(X_pca, Y_val, k) / tf.maximum(tf.math.unsorted_segment_sum(tf.ones_like(X_pca), Y_val, k), 1))

        resultados = list(zip(Y_val, nombres))
        resultados.sort(key=lambda x: x[1])

        # Guardar resultados en archivo
        clases_path = os.path.join(os.path.dirname(csv_path), "clases.txt")
        with open(clases_path, "w") as archivo:
            for clase, nombre in resultados:
                archivo.write(f"{clase} {nombre}\n")
                print(f"Clase {clase} - Archivo {nombre}")
    return clases_path

# Unificación de segmentos por clase
def unificar_clases(clases_file_path, carpeta_segmentos):
    clasescontent = []
    with open(clases_file_path, 'r') as clases_file:
        clasescontent = [int(line.split()[0]) for line in clases_file]

    for clase in range(NUM_MAX_CLASES):
        print(f"Iterando sobre clase {clase}")
        indices_clase = np.where(np.array(clasescontent) == clase)[0]
        print(f"Índices de clase {clase}: {indices_clase}")

        audiototal = []
        sr = None
        for indice in indices_clase:
            nombre_segmento = os.path.join(carpeta_segmentos, f"{os.path.basename(carpeta_segmentos).replace('_mfccs', '')}_{indice:05d}.wav")
            if os.path.exists(nombre_segmento):
                y, sr = librosa.load(nombre_segmento, sr=None)
                audiototal.append(y)

        if audiototal:
            audiototal = np.concatenate(audiototal)
            nombre_salida = os.path.join(carpeta_segmentos, f"{os.path.basename(carpeta_segmentos).replace('_mfccs', '')}_CLASE_{clase}.wav")
            if sr is not None:
                sf.write(nombre_salida, audiototal, sr)
                print(f"Archivo unificado guardado: {nombre_salida}")

# Código principal
def main():
    nombre_archivo = sys.argv[1]

    # Paso 1: Segmentación de audio
    carpeta_segmentos = segmentar_audio(nombre_archivo)

    # Paso 2: Extracción de características y almacenamiento en CSV
    archivo_csv = extraer_caracteristicas(carpeta_segmentos)

    # Paso 3: Clasificación de segmentos
    clases_path = clasificar_segmentos(archivo_csv)

    # Paso 4: Unificación de segmentos por clase
    unificar_clases(clases_path, carpeta_segmentos)

if __name__ == "__main__":
    main()
