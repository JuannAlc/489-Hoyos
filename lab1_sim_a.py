import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, bode

# Define the transfer function H(s) = 1 + (s/1000)
numerator = [1/10000, 1]  # (s/1000) + 1
Denominator = [1]  # FIR filter has no poles apart from origin

# Create transfer function
system = TransferFunction(numerator, Denominator)

# Compute Bode plot
frequencies = np.logspace(1, 6, num=500)  # Frequency range from 10 to 1M Hz
w, mag, phase = bode(system, w=frequencies)

# Plot magnitude response
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.title('Bode Plot of H(s) = 1 + (s/10000)')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude (dB)')
plt.grid(which='both', linestyle='--', linewidth=0.5)

# Plot phase response
plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (degrees)')
plt.grid(which='both', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
