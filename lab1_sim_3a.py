import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Parameters
F = 2e6  # Frequency of the signal (Hz)
Fs = 5e6  # Sampling frequency (Hz)
N = 50  # Number of samples

# Time vector
t = np.arange(N) / Fs

# Signal
x = np.cos(2 * np.pi * F * t)

# Compute DFT using FFT
X = fft(x)

# Frequency axis
freqs = np.fft.fftfreq(N, 1/Fs)

# Plot magnitude spectrum (positive side only)
plt.figure(figsize=(8, 5))
plt.stem(freqs[:N // 2], np.abs(X[:N // 2]))
plt.title('Output (Positive Frequencies)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
