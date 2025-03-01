import numpy as np
import matplotlib.pyplot as plt

fs = 405e6  # Sampling frequency (needed to adjust doesnt work with 400)
f_tone = 200e6  # Tone frequency
n_bits = 6  # 6-bit quantizer
n_levels = 2**n_bits
cycles = 30
samples_per_cycle = fs / f_tone
samples = int(cycles * samples_per_cycle)
t = np.arange(samples) / fs

# Full-scale tone of 200 MHz
sinewave = np.sin(2 * np.pi * f_tone * t)

# Quantization
quantized_wave = np.round((sinewave + 1) * (n_levels / 2 - 1)) / (n_levels / 2 - 1) - 1

# Compute noise power and signal power
quantization_error = sinewave - quantized_wave
quantization_noise_power = np.mean(quantization_error**2)
signal_power = np.mean(sinewave**2)

# SNR for 30 periods
snr = 10 * np.log10(signal_power / quantization_noise_power)

# Ideal SNR for 6-bit quantizer
snr_ideal = 6.02 * n_bits + 1.76


print(f"Ideal SNR (6 bits): {snr_ideal:.2f} dB")
print(f"Empirical SNR (6 bits, exactly 30 periods): {snr:.2f} dB")

# --- PSD Calculation (using DFT) ---

# Compute DFT for 30 periods of quantized signal
X = np.fft.fft(quantized_wave)

# Compute PSD (normalized)
psd = (1 / (samples * fs)) * np.abs(X)**2

# Convert to dB/Hz (avoid log(0) issue by replacing zeros with small value)
psd[psd == 0] = 1e-12
psd_db = 10 * np.log10(psd)

# freq bins
frequencies = np.fft.fftfreq(samples, 1/fs)

# Plot PSD (only positive frequencies)
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:samples//2] / 1e6, psd_db[:samples//2], label='PSD (DFT-based)')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title(f'Power Spectral Density (30 periods)')
plt.grid(True)
plt.ylim(-120, 0)
plt.xlim(0, fs/2 / 1e6)
plt.legend()
plt.show()
