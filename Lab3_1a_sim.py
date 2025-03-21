import numpy as np
import matplotlib.pyplot as plt

freq = 1e9  #1GHz input frequency
A = 2  # Amplitude of 2V
fs = 10e9  # 10GHz sampling frequency
T = 1 / fs  # Sampling period
t_simulation = 5e-9  # 5ns
t = np.linspace(0, t_simulation, 1000)  

# input sine wave genertated
V_in = A * np.sin(2 * np.pi * freq * t)

# sample input signal
sample_times = np.arange(0, t_simulation, T)
V_samples = A * np.sin(2 * np.pi * freq * sample_times)

# simulate ZOH
V_out = np.zeros_like(t)
for i in range(len(sample_times) - 1):
    indices = (t >= sample_times[i]) & (t < sample_times[i + 1])
    V_out[indices] = V_samples[i]

# plot output of sampling
plt.figure(figsize=(10, 5))
plt.plot(t, V_in, label="Input Signal (V_in)", linestyle="dashed", alpha=0.7)
plt.step(t, V_out, where='post', label="Sampled Output (V_out)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid()
plt.show()
