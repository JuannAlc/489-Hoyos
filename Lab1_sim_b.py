import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz, tf2zpk

# Define the FIR filter H(z) = 1 + z^-1 + z^-2 + z^-3 + z^-4
numerator = [1, 1, 1, 1, 1]  # Coefficients for FIR filter
denominator = [1]  # FIR filter has no poles

# Compute frequency response
w, h = freqz(numerator, worN=8000)

# Compute zeros and poles
zeros, poles, gain = tf2zpk(numerator, denominator)

# Print zeros and poles
print("Zeros:", zeros)
print("Poles:", poles)  # Should be empty for an FIR filter

# Plot magnitude response
plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
plt.plot(w / np.pi, 20 * np.log10(abs(h)))
plt.title('Frequency Response of FIR Filter H(z)')
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Magnitude (dB)')
plt.grid()

# Plot phase response
plt.subplot(3, 1, 2)
plt.plot(w / np.pi, np.angle(h, deg=True))
plt.xlabel('Normalized Frequency (×π rad/sample)')
plt.ylabel('Phase (degrees)')
plt.grid()

# Plot zero locations with unit circle
plt.subplot(3, 1, 3)
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', facecolors='none', edgecolors='r', label='Zeros')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Draw unit circle
theta = np.linspace(0, 2 * np.pi, 300)
plt.plot(np.cos(theta), np.sin(theta), 'b--', label='Unit Circle')

plt.grid()
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Pole-Zero Plot')
plt.legend()

plt.tight_layout()
plt.show()
