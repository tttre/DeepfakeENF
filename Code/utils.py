from scipy.io import wavfile
from scipy.signal import butter, sosfilt, resample, savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from RFA import RFA

def read_audio(file_path):
    sampling_rate, data = wavfile.read(file_path)
    return sampling_rate, data

def Bandpass(data,fs,band,order):
    sos = butter(order, band, btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos,data)

def extract_ENF(data_path):
    sampling_rate_old, data = read_audio(data_path)

    sampling_rate = 400.0 
    num_samples = int(len(data) * sampling_rate / sampling_rate_old)
    data_downsampled = resample(data, num_samples) #UNDERSTAND DOWNSAMPLE
    data_filtered = Bandpass(data_downsampled, sampling_rate, [95, 105], 6)


    alpha = 0.25*sampling_rate/(np.max(np.abs(data_filtered)))
    eps = 1e-9
    tau = 750
    I = 1
    ENF_drift = RFA(data_filtered, alpha, tau, eps, I, 1/sampling_rate)
    return ENF_drift, sampling_rate

