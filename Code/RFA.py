from scipy.signal import stft
import numpy as np
from numba import njit
from scipy.signal import hilbert


@njit
def angle_Rz(tau, n, angle_z, Nx):
    idx_plus = int(n + tau)
    idx_minus = int(n - tau)
        
    if 0 <= idx_plus < Nx and 0 <= idx_minus < Nx:
        return angle_z[idx_plus] - angle_z[idx_minus]
    return 0.0

@njit
def angle_Kz(tau, n, angle_z, ts, f, Nx, fs):
    if np.isnan(f) or abs(f) < 1e-3:
        f = 1e-3 if f >= 0 else -1e-3
    tau_prime = int(tau + np.round(fs / (4 * f)))
    a_Rz       = angle_Rz(tau,       n, angle_z, Nx)
    a_Rz_prime = angle_Rz(tau_prime, n, angle_z, Nx)
    weight = ts * f * np.pi
    return (a_Rz       * weight * np.sin(2 * np.pi * ts * f * tau)
          + a_Rz_prime * weight * np.cos(2 * np.pi * ts * f * tau))

def weighted_energy_IF(s,fs,window): 
    print("v2")
    freqs, times, ft = stft(s, fs=fs, nperseg=int(2*fs), noverlap=int(2*fs)-1)
    in_band = (freqs >= window[0]) & (freqs <= window[1])
    freqs_band = freqs[in_band]

    f_est = np.array([
        np.sum(freqs_band * np.abs(ft[in_band, i])) / np.sum(np.abs(ft[in_band, i]))
        for i in range(ft.shape[1])
    ])

    sample_times = times * fs   # scipy's times already account for the padding offset
    return np.interp(np.arange(len(s)), sample_times, f_est,
                     left=f_est[0], right=f_est[-1])

def RFA(data,t,eps,I,ts):
    Nx = len(data)
    x1 = np.copy(data)
    f0 = np.ones(Nx)*100
    fs = 1/ts
    a = (1/4) * fs / np.max(np.abs(data))
    s = np.zeros(Nx)

    for i in range(I):
        angle_z = 2 * np.pi * ts * a * np.cumsum(x1)

        s_hat = np.zeros(Nx)
        for n in range(Nx):
            f = f0[n]
            angle_Kz(3,n,angle_z,ts,f,Nx,fs)
            s_hat[n] = np.sum([angle_Kz(j,n,angle_z,ts,f,Nx,fs) for j in range(t+1)])/((t+1)*t*np.pi*a if t > 0 else 1)

        f_new = weighted_energy_IF(s_hat, fs,[99.5,100.5])

        if (np.sum((f_new-f0)**2))/(np.sum(np.pow(f0,2))) <= eps:
            s = s_hat
            break

        s = s_hat
        f0 = f_new
    return s, f_new
