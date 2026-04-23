from scipy.io import wavfile
from scipy.signal import butter, sosfilt, resample, savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from RFA import RFA,weighted_energy_IF
#----------------------------------------------------------------------------
#Reads the audio file and returns the sampling rate and data array
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
    data_downsampled = resample(data, num_samples)
    data_filtered = Bandpass(data_downsampled, sampling_rate, [95, 105], 6)


    alpha = 0.25*sampling_rate/(np.max(np.abs(data_filtered)))
    eps = 1e-9
    tau = 750
    I = 1
    ENF_drift = RFA(data_filtered, tau, eps, I, 1/sampling_rate)
    return ENF_drift, sampling_rate
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
#AR Process for testing
def generate_AR_process(N):
    mean = 100.0
    std = 0.04
    noise = np.random.normal(0,1,N)
    x = np.zeros(N)
    x[0] = noise[0]
    for n in range(1, N):
        x[n] = 0.99 * x[n-1] + noise[n]
    x = (x - np.mean(x)) / np.std(x) * std + mean
    return x

def generate_noisy_signal(N,fs,SNR):
    f_drift = generate_AR_process(N)
    phase = 2 * np.pi * np.cumsum(f_drift) / fsq
    enf_waveform = np.cos(phase)
    signal_power = np.mean(enf_waveform**2)
    noise_power  = signal_power/(10 ** (SNR/10))
    noise =np.sqrt(noise_power)*np.random.normal(0, 1, N)
    noisy_ENF = enf_waveform + noise
    return noisy_ENF, f_drift

def extract_ENF_from_test(N=10000, SNR=-20.0):
    noisy_signal, ground = generate_noisy_signal(N,400.0,SNR)
    denoised_s, f_est = RFA(noisy_signal, t=750, eps=1e-9, I=3, ts=1/400.0)
    f_est_dev  = f_est  - np.mean(f_est)
    ground_dev = ground - np.mean(ground)
    f_baseline = weighted_energy_IF(noisy_signal, 400.0, [99.5, 100.5])
    def metrics(f_hat, f_gt, label):
        d = f_hat - np.mean(f_hat)
        g = f_gt  - np.mean(f_gt)
        nm = np.sum((d - g)**2) / np.sum(g**2)
        cc = np.corrcoef(d, g)[0, 1]
        print(f"{label:20s}  NM={nm:.4f}  CC={cc:.4f}")

    metrics(f_baseline, ground, "Baseline (no RFA)")
    metrics(f_est,      ground, "RFA output")
    plt.plot(ground, label='Groundtruth ENF')
    plt.plot(f_est, label='Denoised ENF')
    plt.legend()
    plt.show()
    return denoised_s, f_est

extract_ENF_from_test(30000,-20)