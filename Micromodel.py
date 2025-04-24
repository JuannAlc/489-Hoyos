import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# Simulate input signal
N = 5000
x = np.linspace(-1, 1, N)
analog_input = 0.9 * np.sin(2 * np.pi * 50 * x)

# Generate bitwise comparator output
def generate_comparator_bits(signal, bits=10):
    scaled = ((signal + 1) / 2 * (2**bits - 1)).astype(int)
    return ((scaled[:, None] & (1 << np.arange(bits))) > 0).astype(int)

# Simulate distorted outputs of a 2-channel (A and B) ADC
def generate_adc_channels(signal):
    ideal = ((signal + 1) / 2) * (2**10 - 1)
    distortion = (
        15 * np.sin(2 * np.pi * 3 * x) +
        10 * np.sin(2 * np.pi * 5 * x) +
        8 * (signal**3) +
        5 * (signal**5)
    )
    chA = ideal + distortion
    chB = ideal - distortion
    return chA, chB

# Prepare data
channelA, channelB = generate_adc_channels(analog_input)
bitwise_input = generate_comparator_bits(analog_input, bits=10)

# Train ML model to minimize error between calibrated A and channel B in the split ADC
errors = []
model = MLPRegressor(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=1, warm_start=True)

for epoch in range(100):  # simulate training 
    model.fit(bitwise_input, channelB)  # predict B using A's bitwise input
    calibrated_output = model.predict(bitwise_input)
    error = np.mean(np.abs(calibrated_output - channelB))  # mean absolute error
    errors.append(error)

# Plot the decreasing error trend after training and adjustign tyhe distortion
plt.figure(figsize=(10, 5))
plt.plot(errors, label="|Calibrated A - B|", color='blue')
plt.xlabel("Training Iteration")
plt.ylabel("Channel-to-Channel Error")
plt.title("ML Calibration Reduces Error Between Channel A and B")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
