import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as welch
from scipy.fftpack import fft

tone_freq = 2e6
A = 1
Fs = 5e6
snr_dB = 50 # given SNR of signal
T = 1e-3  
t = np.arange(0, T, 1/Fs)

signal = A * np.sin(2*np.pi*tone_freq*t)
snr = 10**(snr_dB/10) # SNR in linear
signal_power = (A**2) / 2  # Power of sine  = A^2/2
variance = signal_power / snr 
noise = np.random.normal(0,np.sqrt(variance/2),len(t))
signal_noise = signal + noise
print(np.average(np.abs(noise)))

# Plot the tone signal and noise
plt.figure(figsize=(10, 5))
plt.plot(t[:25], signal_noise[:25], alpha=0.75)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.title("Tone transient")
plt.legend()
plt.grid()
plt.show()

# Plot noise
plt.figure(figsize=(10, 5))
plt.plot(t[:100], noise[:100], color="r", alpha=0.75)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.title("Noise Plotted")
plt.legend()
plt.grid()
plt.show()

# Compute the DFT of the noisy sampled signal
N = len(signal_noise)  # Number of samples
frequencies_dft = np.fft.fftfreq(N, 1/Fs)  # Frequency axis
dft_signal = fft(signal_noise)
psd_dft = (np.abs(dft_signal) ** 2) / N  # Compute PSD from DFT
frequencies_dft = np.fft.fftfreq(N, 1/Fs)  # Frequency axis

# Normalize PSD so that the signal peak is at 0 dB
psd_dft_dB = 10 * np.log10(psd_dft)
psd_dft_dB_normalized = psd_dft_dB - np.max(psd_dft_dB)

# Plot the DFT of the noisy signal
plt.figure(figsize=(10, 5))
plt.plot(frequencies_dft[:N//2], 20 * np.log10(np.abs(dft_signal[:N//2])))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('DFT of Signal with Gaussian Noise')
plt.grid()
plt.show()

# Plot the Power Spectral Density (PSD) from the DFT
plt.figure(figsize=(10, 5))
plt.plot(frequencies_dft[:N//2], psd_dft_dB_normalized[:N//2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB/Hz)')
plt.title('PSD Plot from DFT (at 0 dB)')
plt.ylim(-50, 10)  # Ensure visibility of the noise floor
plt.grid()
plt.show()