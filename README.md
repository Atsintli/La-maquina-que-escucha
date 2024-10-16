# Maestro de Análisis de Audio

Este script permite realizar un análisis avanzado de archivos de audio, incluyendo segmentación, extracción de características con MFCC, clasificación mediante PCA y k-Means utilizando TensorFlow, y unificación de segmentos por clase.

## About The Project

Este proyecto proporciona un script de Python para analizar archivos de audio. Realiza las siguientes funciones:

* **Segmentación:** Divide el audio en segmentos basados en la detección de onsets.
* **Extracción de características:** Extrae coeficientes MFCC (Mel-Frequency Cepstral Coefficients) de cada segmento.
* **Clasificación:**  Utiliza PCA (Principal Component Analysis) para reducir la dimensionalidad de las características y k-Means para agrupar los segmentos en clases.
* **Unificación:** Combina los segmentos de audio que pertenecen a la misma clase.

### Built With

* [Python 3.x](https://www.python.org/)
* [Librosa](https://librosa.org/doc/latest/index.html)
* [NumPy](https://numpy.org/)
* [SoundFile](https://pysoundfile.readthedocs.io/en/latest/)
* [Matplotlib](https://matplotlib.org/)
* [Pandas](https://pandas.pydata.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)


## Getting Started

Para utilizar este script, sigue los siguientes pasos:

### Prerequisites

Asegúrate de tener Python 3.x instalado en tu sistema.

### Installation

1. Clona el repositorio (si aplica).
2. Instala las dependencias necesarias:

   ```bash
   pip install librosa numpy soundfile matplotlib pandas tensorflow scikit-learn

   ## 2. **Audio Segmentation**
The script divides the audio file into segments based on detected onset peaks. These segments are saved as separate `.wav` files in a new directory.

- **Function**: `segmentar_audio(nombre_archivo)`
- **Libraries**: `librosa`, `soundfile`
- **Process**:
  - Creates a directory for segments.
  - Loads the audio file.
  - Detects onset peaks using the `librosa.onset.onset_strength()` function.
  - Converts the detected peaks into time samples and segments the audio.
  - Saves each segment as a `.wav` file.

## 5. **Class Unification**
The script merges all audio segments belonging to the same class into a single audio file for each class.

- **Function**: `unificar_clases(clases_file_path, carpeta_segmentos)`
- **Libraries**: `numpy`, `librosa`, `soundfile`, `os`
- **Process**:
  - Reads the class assignments from the text file.
  - For each class, it groups the corresponding audio segments.
  - Concatenates the audio segments that belong to the same class into a single audio stream.
  - Saves the concatenated audio for each class as a `.wav` file in the segments folder.

## 6. **Main Execution**
The script sequentially executes the steps necessary to process and classify audio segments into distinct classes, merging those that belong to the same class into new audio files.

- **Function**: `main()`
- **Libraries**: `sys`
- **Process**:
  - Takes the input audio file as an argument from the command line.
  - Calls the segmentation function to split the audio file into segments.
  - Extracts the MFCC features for each segment and stores them in a CSV file.
  - Classifies the segments using K-means clustering.
  - Merges segments belonging to the same class into one audio file per class.

```python
def main():
    nombre_archivo = sys.argv[1]
    carpeta_segmentos = segmentar_audio(nombre_archivo)
    archivo_csv = extraer_caracteristicas(carpeta_segmentos)
    clases_file_path = clasificar_segmentos(archivo_csv)
    unificar_clases(clases_file_path, carpeta_segmentos)

if __name__ == "__main__":
    main()
