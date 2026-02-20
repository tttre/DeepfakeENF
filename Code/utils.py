from scipy.io import wavfile
from scipy.signal import butter, sosfilt, resample
import matplotlib.pyplot as plt
import numpy as np
from RFA import RFA

def read_audio(file_path):
    sampling_rate, data = wavfile.read(file_path)
    return sampling_rate, data

def Bandpass(data,fs,band,order):
    sos = butter(order, band, btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos,data)


sampling_rate_old, data = read_audio("/Users/ciaranwehrli/Desktop/DeepfakeENF/synthetic_enf_minus20dB.wav")

sampling_rate = 400.0 
num_samples = int(len(data) * sampling_rate / sampling_rate_old)
data_downsampled = resample(data, num_samples) #UNDERSTAND DOWNSAMPLE
data_filtered = Bandpass(data_downsampled, sampling_rate, [95, 105], 6)


alpha = 0.25*sampling_rate/(np.max(np.abs(data_filtered)))
print(alpha)
eps = 0.01
tau = 750
I = 3

ENF_drift = RFA(data_filtered, alpha, tau, eps, 3, 1/sampling_rate)

time_axis = np.linspace(0, len(ENF_drift) / sampling_rate, len(ENF_drift))
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(time_axis, ENF_drift, color='b', label='Extracted ENF')
ax.axhline(y=100.0, color='r', linestyle='--', alpha=0.7, label='100 Hz Base')
ax.set_title("ENF Drift")
ax.set_xlabel("Time")
ax.set_ylabel("Frequency")
ax.grid(True)
ax.legend()
plt.show()

