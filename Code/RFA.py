from scipy.signal import stft
import numpy as np
from numba import njit

@njit
def angle_Rz(tau, n, angle_z, Nx):
    idx_plus = int(n + tau)
    idx_minus = int(n - tau)
        
    if 0 <= idx_plus < Nx and 0 <= idx_minus < Nx:
        return angle_z[idx_plus] - angle_z[idx_minus]
    return 1.0 + 0j

@njit
def angle_Kz(tau,n,angle_z,ts,f,Nx,fs):
    if np.isnan(f) or abs(f) < 1e-3:
        f = 1e-3 if f >= 0 else -1e-3
    tau_prime = int(tau + np.round(fs/(4*f)))
    a_Rz = angle_Rz(tau,n,angle_z,Nx)
    a_Rz_prime = angle_Rz(tau_prime,n,angle_z,Nx)
    return a_Rz*ts*f*tau*np.pi*np.sin(2*np.pi*ts*f*tau)+a_Rz_prime*(ts*f*tau*np.pi*np.cos(2*np.pi*ts*f*tau))

def weighted_energy_IF(s,fs,window): 
    freqs, times, ft = stft(s, fs=fs, nperseg=int(2*fs), noverlap=int(2*fs)-1)
    freqs_new = freqs[(window[0] <= freqs) & (freqs <= window[1])]
    f_est = []
    for i in range(ft.shape[1]):
        mags = np.abs([ft[n,i] for n in range(len(freqs)) if window[0] <= freqs[n] <= window[1]])
        f_est.append(np.sum(freqs_new*mags)/np.sum(mags))
    return np.interp(np.arange(len(s)), np.linspace(0, len(s), len(f_est)), f_est) #UNDERSTAND INTERPOLATION AND STFT

def RFA(data,a,t,eps,I,ts):
    Nx = len(data)
    print("Nx: ", Nx)
    x1 = np.copy(data)
    f0 = np.ones(Nx)*100
    fs = 1/ts
    s = np.zeros(Nx)

    for i in range(I):
        print("Iteration i: ", i)
        angle_z = 2 * np.pi * ts * a * np.cumsum(x1)

        s_hat = np.zeros(Nx)
        for n in range(Nx):
            f = f0[n]
            print(angle_Kz(3,n,angle_z,ts,f,Nx,fs))
            s_hat[n] = np.sum([angle_Kz(j,n,angle_z,ts,f,Nx,fs) for j in range(t+1)])/((t+1)*t*np.pi*a if t > 0 else 1)
            if n%100 == 0:
                print("Inner Iteration: ", n)

        f_new = weighted_energy_IF(s_hat, fs,[99.5,100.5])

        if (np.sum((f_new-f0)**2))/(np.sum(np.pow(f0,2))) <= eps:
            s = s_hat
            break

        s = s_hat
        f0 = f_new

    return f_new
