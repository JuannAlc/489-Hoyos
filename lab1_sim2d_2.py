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

# Sampling times for case 1 (0 : Ts : T - Ts)
n_samples_case_1 = np.arange(0, T, Ts)
x_samples_case_1 = np.cos(2 * np.pi * F1 * n_samples_case_1)

# Sampling times for case 2 (Ts/2 : Ts : T - Ts/2)
n_samples_case_2 = np.arange(Ts / 2, T, Ts)
x_samples_case_2 = np.cos(2 * np.pi * F1 * n_samples_case_2)

# Reconstruct the signal using sinc interpolation
def sinc_interpolation(t, n_samples, x_samples, Ts):
    x_r = np.zeros_like(t)
    for n, x_n in zip(n_samples, x_samples):
        x_r += x_n * np.sinc((t - n) / Ts)  # sinc interpolation
    return x_r

# Reconstruct the signals for both cases
x_r_t_case_1 = sinc_interpolation(t_cont, n_samples_case_1, x_samples_case_1, Ts)
x_r_t_case_2 = sinc_interpolation(t_cont, n_samples_case_2, x_samples_case_2, Ts)

# Calculate Mean Squared Error (MSE) for both cases
mse_case_1 = np.mean((x_r_t_case_1 - x1_t)**2)
mse_case_2 = np.mean((x_r_t_case_2 - x1_t)**2)

print(f'Mean Squared Error (MSE) for Case 1 (0 : Ts : T - Ts): {mse_case_1:.6e}')
print(f'Mean Squared Error (MSE) for Case 2 (Ts/2 : Ts : T - Ts/2): {mse_case_2:.6e}')

# Plot the results
plt.figure(figsize=(12, 8))

# Plot original continuous signal
plt.plot(t_cont, x1_t, label='Original signal x1(t)', linewidth=2)

# Plot reconstructed signals
plt.plot(t_cont, x_r_t_case_1, label='Reconstructed signal (Case 1)', linestyle='dashed')
plt.plot(t_cont, x_r_t_case_2, label='Reconstructed signal (Case 2)', linestyle='dotted')

# Mark the sampled points for both cases
plt.scatter(n_samples_case_1, x_samples_case_1, color='red', label='Sampled points (Case 1)', zorder=5)
plt.scatter(n_samples_case_2, x_samples_case_2, color='green', label='Sampled points (Case 2)', zorder=5)

plt.title('Signal Reconstruction with Sinc Interpolation')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
