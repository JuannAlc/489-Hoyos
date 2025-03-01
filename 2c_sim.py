import numpy as np
import matplotlib.pyplot as plt

def compute_snr(n_bits, fs=405e6, f_tone=200e6, cycles=100):
    n_levels = 2**n_bits
    samples_per_cycle = fs / f_tone
    samples = int(cycles * samples_per_cycle)
    t = np.arange(samples) / fs
    
    sinewave = np.sin(2 * np.pi * f_tone * t)
    quantized_wave = np.round((sinewave + 1) * (n_levels / 2 - 1)) / (n_levels / 2 - 1) - 1
    
    quantization_error = sinewave - quantized_wave
    quantization_noise_power = np.mean(quantization_error**2)
    signal_power = np.mean(sinewave**2)
    snr = 10 * np.log10(signal_power / quantization_noise_power)
    snr_ideal = 6.02 * n_bits + 1.76
    
    print(f"Ideal SNR ({n_bits} bits): {snr_ideal:.2f} dB")
    print(f"Empirical SNR ({n_bits} bits, exactly 30 periods): {snr:.2f} dB")
    
    return quantized_wave, samples, fs

def plot_psd(signal, samples, fs, n_bits):
    X = np.fft.fft(signal)
    psd = (1 / (samples * fs)) * np.abs(X)**2
    psd[psd == 0] = 1e-12
    psd_db = 10 * np.log10(psd)
    frequencies = np.fft.fftfreq(samples, 1/fs)
    
    plt.plot(frequencies[:samples//2] / 1e6, psd_db[:samples//2], label=f'PSD {n_bits}-bit')

plt.figure(figsize=(10, 6))
for n_bits in [6, 12]:
    quantized_wave, samples, fs = compute_snr(n_bits)
    plot_psd(quantized_wave, samples, fs, n_bits)
    
plt.xlabel('Frequency (MHz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Power Spectral Density (30 periods)')
plt.grid(True)
plt.ylim(-120, 0)
plt.xlim(0, fs/2 / 1e6)
plt.legend()
plt.show()
