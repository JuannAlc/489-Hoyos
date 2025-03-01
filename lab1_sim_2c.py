import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz

# FIR Filter Example
# Define FIR filter coefficients (e.g., a simple moving average filter)
fir_coeffs = np.ones(5) / 5  # Moving average filter of length 5

# Generate an impulse input
n_samples = 50
impulse = np.zeros(n_samples)
impulse[0] = 1  # Impulse at n=0

# Apply FIR filter
fir_response = lfilter(fir_coeffs, [1], impulse)

# Plot FIR response
plt.figure(figsize=(10, 6))
plt.stem(fir_response, basefmt=" ", use_line_collection=True)
plt.title("FIR Filter Impulse Response")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# IIR Filter Example
# Define IIR filter coefficients (e.g., a first-order filter)
iir_coeffs_b = [1]          # Numerator coefficients
iir_coeffs_a = [1, -0.9]   # Denominator coefficients (pole at 0.9)

# Apply IIR filter
iir_response = lfilter(iir_coeffs_b, iir_coeffs_a, impulse)

# Plot IIR response
plt.figure(figsize=(10, 6))
plt.stem(iir_response, basefmt=" ", use_line_collection=True)
plt.title("IIR Filter Impulse Response (Stable)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Unstable IIR Filter Example
# Define IIR filter with a pole outside the unit circle
iir_coeffs_a_unstable = [1, -1.1]  # Denominator coefficients (pole at 1.1)

# Apply unstable IIR filter
iir_response_unstable = lfilter(iir_coeffs_b, iir_coeffs_a_unstable, impulse)

# Plot unstable IIR response
plt.figure(figsize=(10, 6))
plt.stem(iir_response_unstable, basefmt=" ", use_line_collection=True)
plt.title("IIR Filter Impulse Response (Unstable)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()
