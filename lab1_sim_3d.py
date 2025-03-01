import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Parameters
F = 2e6  # Frequency of x(t) signal (Hz)
F1 = 200e6  # Frequency of the first component (Hz)
F2 = 400e6  # Frequency of the second component (Hz)
Fs = 1e9  # Sampling frequency (Hz)
N = 50  # Number of samples

# Time vector
t = np.arange(N) / Fs

# Signal x(t)
x = np.cos(2 * np.pi * F * t)

# Apply Blackman window to x(t)
window = np.blackman(N)
x_windowed = x * window

# Compute DFT using FFT for x(t)
X_windowed = fft(x_windowed)

# Signal y(t)
y = np.cos(2 * np.pi * F1 * t) + np.cos(2 * np.pi * F2 * t)

# Apply Blackman window to y(t)
y_windowed = y * window

# Compute DFT using FFT for y(t)
Y_windowed = fft(y_windowed)

# Frequency axis
freqs = np.fft.fftfreq(N, 1/Fs)

# Plot magnitude spectrum (positive side only) for x(t)
plt.figure(figsize=(8, 5))
plt.stem(freqs[:N // 2], np.abs(X_windowed[:N // 2]))
plt.title('Magnitude Spectrum of x(t) with Blackman Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()

# Plot magnitude spectrum (positive side only) for y(t)
plt.figure(figsize=(8, 5))
plt.stem(freqs[:N // 2], np.abs(Y_windowed[:N // 2]))
plt.title('Magnitude Spectrum of y(t) with Blackman Window')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
