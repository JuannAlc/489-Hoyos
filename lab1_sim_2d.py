import numpy as np
import matplotlib.pyplot as plt

# Define constants
F1 = 300e6  # Frequency of the cosine wave (300 MHz)
Fs = 500e6  # Sampling frequency (800 MHz)
Ts = 1 / Fs  # Sampling period
T = 10 / F1  # Duration of 10 cycles of the cosine wave

# Time vector for the original signal (continuous signal)
t_cont = np.linspace(0, T, 1000)

# Define the original signal x1(t) = cos(2*pi*F1*t)
x1_t = np.cos(2 * np.pi * F1 * t_cont)

# Sampling times (discrete times)
n_samples = np.arange(0, T, Ts)
x_samples = np.cos(2 * np.pi * F1 * n_samples)

# Reconstruct the signal using sinc interpolation
def sinc_interpolation(t, n_samples, x_samples, Ts):
    x_r = np.zeros_like(t)
    for n, x_n in zip(n_samples, x_samples):
        x_r += x_n * np.sinc((t - n) / Ts)  # sinc interpolation
    return x_r

# Reconstruct the signal at the sample points (for comparison)
x_r_t = sinc_interpolation(t_cont, n_samples, x_samples, Ts)

# Calculate Mean Squared Error (MSE) on sampled points
mse = np.mean((x_r_t - x1_t)**2)
print(f'Mean Squared Error (MSE): {mse:.6e}')

# Plot the results
plt.figure(figsize=(10, 6))

# Plot original continuous signal
plt.plot(t_cont, x1_t, label='Original signal x1(t)', linewidth=2)

# Plot reconstructed signal
plt.plot(t_cont, x_r_t, label='Reconstructed signal x_r(t)', linestyle='dashed')

# Mark the sampled points
plt.scatter(n_samples, x_samples, color='red', label='Sampled points', zorder=5)

plt.title('Signal Reconstruction with Sinc Interpolation')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
