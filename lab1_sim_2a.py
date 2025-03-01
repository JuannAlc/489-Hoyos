import numpy as np
import matplotlib.pyplot as plt

# Given frequencies
F1 = 300e6  # 300 MHz
F2 = 800e6  # 800 MHz
Fs = 500e6  # 500 MHz (Sampling frequency)

# Time vector for continuous signals (0 to 10 ns)
t_cont = np.linspace(0, 10e-9, 1000)  # 10 ns duration

# Define continuous signals
x1_t = np.cos(2 * np.pi * F1 * t_cont)
x2_t = np.cos(2 * np.pi * F2 * t_cont)

# Sample points (discrete time indices)
n = np.arange(0, 10)  # Up to 10 samples within 10 ns
t_sampled = n / Fs  # Corresponding sampled time

# Sampled signals
x1_n = np.cos(2 * np.pi * F1 * t_sampled)
x2_n = np.cos(2 * np.pi * F2 * t_sampled)

# Plot continuous-time signals with sampled data as solid lines
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t_cont * 1e9, x1_t, label="x1(t) = cos(2πF1t)", color='blue')
plt.plot(t_sampled * 1e9, x1_n, '-o', label="x1[n] (sampled)", color='orange', markersize=6)
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.title("Sampling of signal x1(t)")
plt.xlim(0, 10)  # Stop at 10 ns
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t_cont * 1e9, x2_t, label="x2(t) = cos(2πF2t)", color='blue')
plt.plot(t_sampled * 1e9, x2_n, '-o', label="x2[n] (sampled)", color='orange', markersize=6)
plt.xlabel("Time (ns)")
plt.ylabel("Amplitude")
plt.title("Sampling of signal x2(t)")
plt.xlim(0, 10)  # Stop at 10 ns
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()