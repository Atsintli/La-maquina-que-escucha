# Matriz de correlación

import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys

# Cargar archivo de audio
audioname = sys.argv[1]
y, sr = librosa.load(audioname, sr=None)

# Calcular MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)

# Matrices de recurrencia con diferentes configuraciones
R1 = librosa.segment.recurrence_matrix(mfcc)
R2 = librosa.segment.recurrence_matrix(mfcc, k=5)
R3 = librosa.segment.recurrence_matrix(mfcc, width=7)
R4 = librosa.segment.recurrence_matrix(mfcc, metric='cosine')
R5 = librosa.segment.recurrence_matrix(mfcc, sym=True)
R6 = librosa.segment.recurrence_matrix(mfcc, mode='affinity')

# Graficar las matrices de recurrencia
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
librosa.display.specshow(R1, x_axis='time', y_axis='time', cmap='gray_r')
plt.title('Recurrencia binaria (simétrica)')

plt.subplot(2, 3, 2)
librosa.display.specshow(R2, x_axis='time', y_axis='time', cmap='gray_r')
plt.title('Recurrencia binaria (k=5)')

plt.subplot(2, 3, 3)
librosa.display.specshow(R3, x_axis='time', y_axis='time', cmap='gray_r')
plt.title('Recurrencia binaria (ancho=7)')

plt.subplot(2, 3, 4)
librosa.display.specshow(R4, x_axis='time', y_axis='time', cmap='gray_r')
plt.title('Recurrencia con similitud coseno')

plt.subplot(2, 3, 5)
librosa.display.specshow(R5, x_axis='time', y_axis='time', cmap='gray_r')
plt.title('Recurrencia mutua')

plt.subplot(2, 3, 6)
librosa.display.specshow(R6, x_axis='time', y_axis='time', cmap='magma_r')
plt.title('Recurrencia con afinidad')

plt.tight_layout()
plt.show()
