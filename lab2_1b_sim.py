import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as welch
from scipy.fftpack import fft
from scipy.signal import get_window

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



# Compute the DFT of the noisy sampled signal
N = len(signal_noise)  # Number of samples
frequencies_dft = np.fft.fftfreq(N, 1/Fs)  # Frequency axis
dft_signal = fft(signal_noise)
psd_dft = (np.abs(dft_signal) ** 2) / N  # Compute PSD from DFT
frequencies_dft = np.fft.fftfreq(N, 1/Fs)  # Frequency axis

# Normalize PSD so that the signal peak is at 0 dB
psd_dft_dB = 10 * np.log10(psd_dft)
psd_dft_dB_normalized = psd_dft_dB - np.max(psd_dft_dB)

# Increase resolution by using a zero-padding approach
N_high_res = 4 * N  # Zero-padding to increase resolution

windows = {
    "Hanning": get_window("hann", N),
    "Hamming": get_window("hamming", N),
    "Blackman": get_window("blackman", N)
}

for name, window in windows.items():
    # Apply window
    signal_windowed = signal_noise * window

    # Compute DFT with higher resolution
    dft_signal = fft(signal_windowed, N_high_res)
    psd_dft = (np.abs(dft_signal) ** 2) / N  # Compute PSD from DFT
    frequencies_dft = np.fft.fftfreq(N_high_res, 1/Fs)  # Frequency axis

    # Normalize PSD so that the signal peak is at 50 dB and noise floor at 0 dB
    psd_dft_dB = 10 * np.log10(psd_dft)
    psd_dft_dB_normalized = psd_dft_dB - np.max(psd_dft_dB) + snr_dB

    # Plot PSD
    plt.figure(figsize=(12, 6), dpi=150)  # Higher resolution
    plt.plot(frequencies_dft[:N_high_res//2], psd_dft_dB_normalized[:N_high_res//2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (dB/Hz)')
    plt.title(f'Power Spectral Density with {name} Window')
    plt.grid()
    plt.show()