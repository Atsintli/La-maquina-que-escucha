Maestro de Análisis de Audio
Este script permite realizar un análisis avanzado de archivos de audio, incluyendo segmentación, extracción de características con MFCC, clasificación mediante PCA y k-Means utilizando TensorFlow, y unificación de segmentos por clase.

Requisitos
Python 3.x
Paquetes:
librosa
numpy
soundfile
matplotlib
pandas
tensorflow
scikit-learn

Instalación de Dependencias
Instalar los paquetes necesarios:

bash
pip install librosa numpy soundfile matplotlib pandas tensorflow scikit-learn

Uso
Ejecuta el script con la siguiente estructura:

bash
python3 LMQE.py <ruta_al_archivo_de_audio>

Segmentación de Audio
El script segmenta el archivo de audio en fragmentos basados en la detección de onsets.

python
import librosa

y, sr = librosa.load(<ruta_al_archivo_de_audio>)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

Extracción de Características
Se extraen las características MFCC para cada segmento y se guardan en un archivo CSV.

python
import numpy as np
import pandas as pd

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
df = pd.DataFrame(mfccs.T)
df.to_csv('caracteristicas_mfcc.csv', index=False)

Clasificación de Segmentos
Se realiza la reducción de dimensionalidad mediante PCA y se agrupan los segmentos utilizando k-Means.

python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Reducción con PCA
pca = PCA(n_components=2)
mfccs_pca = pca.fit_transform(mfccs.T)

# Clasificación con k-Means
kmeans = KMeans(n_clusters=3)
kmeans.fit(mfccs_pca)
clases = kmeans.predict(mfccs_pca)
np.savetxt('clases.txt', clases, fmt='%d')

Unificación de Segmentos
Los segmentos de audio clasificados se combinan y se exportan en archivos separados por clase.

python
import soundfile as sf

for i, clase in enumerate(np.unique(clases)):
    clase_indices = np.where(clases == clase)[0]
    segmentos = [y[onset_frames[i]:onset_frames[i+1]] for i in clase_indices]
    audio_unido = np.concatenate(segmentos)
    sf.write(f'segmento_clase_{clase}.wav', audio_unido, sr)

Ejemplo de Uso
Instalación de dependencias:
bash
pip install librosa numpy soundfile matplotlib pandas tensorflow scikit-learn

Ejecución del análisis:
bash
python3 LMQE.py archivo_audio.wav

Esto genera:
caracteristicas_mfcc.csv: CSV con los valores MFCC de cada segmento.
clases.txt: Archivo de texto con la clasificación de los segmentos.
Archivos WAV por clase: Archivos de audio unificados por clase.
