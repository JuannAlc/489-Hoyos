import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Parameters
F1 = 200e6  # Frequency of the first component (Hz)
F2 = 400e6  # Frequency of the second component (Hz)
Fs = 500e6  # Sampling frequency (Hz)
N = 50  # Number of samples

# Time vector
t = np.arange(N) / Fs

# Signal
y = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)

# Compute DFT using FFT
Y = fft(y)

# Frequency axis
freqs = np.fft.fftfreq(N, 1/Fs)

# Plot magnitude spectrum (positive side only)
plt.figure(figsize=(8, 5))
plt.stem(freqs[:N // 2], np.abs(Y[:N // 2]))
plt.title('Output (Positive Frequencies)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()