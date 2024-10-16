# LMQE

This script enables advanced analysis of audio files, including segmentation, feature extraction with MFCC, classification using PCA and k-Means with TensorFlow, and merging of segments by class.

## About The Project

This project provides a Python script for analyzing audio files. It performs the following functions:

* **Segmentation:** Divides the audio into segments based on onset detection.
* **Feature Extraction:** Extracts MFCC (Mel-Frequency Cepstral Coefficients) from each segment.
* **Classification:** Utilizes PCA (Principal Component Analysis) to reduce the dimensionality of the features and k-Means to group the segments into classes.
* **Merging:** Combines audio segments that belong to the same class.


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

1. **Clonar el Repositorio**

   ```bash
   git clone https://github.com/tu-usuario/maestro-analisis-audio.git
   cd maestro-analisis-audio

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
  
## 3. **Feature Extraction and CSV Storage**
The script extracts Mel-frequency cepstral coefficients (MFCC) from each audio segment and saves them in a CSV file.

- **Function**: `extraer_caracteristicas(carpeta_segmentos)`
- **Libraries**: `librosa`, `csv`, `os`
- **Process**:
  - Creates a new directory for storing MFCC files.
  - Loads each `.wav` file from the segments folder.
  - Extracts MFCCs using `librosa.feature.mfcc()` function.
  - Saves the extracted MFCC features to a CSV file, with rows representing each audio segment.

## 4. **Segment Classification**
The script applies K-means clustering to classify audio segments based on the MFCC features. Before clustering, it reduces the data dimensionality using PCA (Principal Component Analysis).

- **Function**: `clasificar_segmentos(csv_path)`
- **Libraries**: `pandas`, `sklearn` (for PCA and KMeans)
- **Process**:
  - Loads the CSV file containing MFCC features.
  - Scales the feature data using `StandardScaler()`.
  - Reduces dimensionality using PCA, retaining enough components to explain the variance effectively.
  - Applies K-means clustering to the PCA-transformed data.
  - Saves the segment class labels in a text file.

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
