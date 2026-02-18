from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt
import numpy as np

def read_audio(file_path):
    sampling_rate, data = wavfile.read(file_path)
    return sampling_rate, data

sampling_rate, data = read_audio("/Users/ciaranwehrli/Desktop/DeepfakeENF/test.wav")
data_transformed = stft(data,fs=sampling_rate,nperseg=16384)
fig, axes = plt.subplots(3, 3, figsize=(15, 10)) # Create 3x3 grid
axes = axes.flatten() # Flatten 2D array of axes to 1D for easy indexing

# Loop to plot 9 different windows
for i in range(9):
    # Select a window (e.g., window 120, 121, 122...)
    window_index = i+3
    
    axes[i].plot(data_transformed[0], np.abs(data_transformed[2][:, window_index]))
    axes[i].set_xlim(0, 300) # Zoom to ENF range
    axes[i].axvline(x=50, color='r', linestyle='--', alpha=0.5) # Swiss 50Hz marker
    axes[i].set_title(f"Window {window_index}")

plt.tight_layout() # Fixes overlapping labels
plt.show()