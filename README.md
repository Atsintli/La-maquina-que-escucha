# Maestro de Análisis de Audio

Este script permite realizar un análisis avanzado de archivos de audio, incluyendo segmentación, extracción de características con MFCC, clasificación mediante PCA y k-Means utilizando TensorFlow, y unificación de segmentos por clase.

## Table of Contents

* [About The Project](#about-the-project)
    * [Built With](#built-with)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
* [Usage](#usage)
    * [Segmentación de Audio](#segmentación-de-audio)
    * [Extracción de Características](#extracción-de-características)
    * [Clasificación de Segmentos](#clasificación-de-segmentos)
    * [Unificación de Segmentos](#unificación-de-segmentos)
* [Example](#example)


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
