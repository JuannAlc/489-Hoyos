import matplotlib.pyplot as plt
import numpy as np

# Define input voltage range
vin = np.linspace(0, 2.5, 1000)

# Transfer functions:
# 2-bit (no redundancy)
thresholds_2b = [0.67, 1.33, 2.0]
outputs_2b = [0.0, 0.67, 1.33, 2.0]

def transfer_2b(v):
    if v < thresholds_2b[0]: return outputs_2b[0]
    elif v < thresholds_2b[1]: return outputs_2b[1]
    elif v < thresholds_2b[2]: return outputs_2b[2]
    else: return outputs_2b[3]

# 2.5-bit without redundancy 
thresholds_2_5b_no_red = [0.4, 0.8, 1.2, 1.6, 2.0]
outputs_2_5b_no_red = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]

def transfer_2_5b_no_red(v):
    for i, t in enumerate(thresholds_2_5b_no_red):
        if v < t:
            return outputs_2_5b_no_red[i]
    return outputs_2_5b_no_red[-1]

# 2.5-bit with redundancy
thresholds_2_5b_red = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
outputs_2_5b_red = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

def transfer_2_5b_red(v):
    for i, t in enumerate(thresholds_2_5b_red):
        if v < t:
            return outputs_2_5b_red[i]
    return outputs_2_5b_red[-1]

# transfer functions
y_2b = [transfer_2b(v) for v in vin]
y_2_5b_no_red = [transfer_2_5b_no_red(v) for v in vin]
y_2_5b_red = [transfer_2_5b_red(v) for v in vin]

# Plotting
plt.figure(figsize=(12, 7))
plt.plot(vin, y_2_5b_red, label='2.5-bit with redundancy', color='blue')
plt.plot(vin, y_2_5b_no_red, label='2.5-bit without redundancy', color='green')
plt.plot(vin, y_2b, label='2-bit', color='red')

# Threshold lines
for t in thresholds_2_5b_red:
    plt.axvline(x=t, color='blue', linestyle=':', linewidth=1)
for t in thresholds_2_5b_no_red:
    plt.axvline(x=t, color='green', linestyle=':', linewidth=1)
for t in thresholds_2b:
    plt.axvline(x=t, color='red', linestyle=':', linewidth=1)

# Labels and legends
plt.title("Transfer Functions")
plt.xlabel("Input Voltage (V)")
plt.ylabel("Output Voltage (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
