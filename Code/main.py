from scipy.io import wavfile
from scipy.signal import butter, sosfilt, resample, savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from utils import extract_ENF

### Extraction and Save
ENF_drift, sampling_rate = extract_ENF("/Users/ciaranwehrli/Desktop/DeepfakeENF/test.wav")

ENF_save = {
    'drift': ENF_drift,
    'fs': sampling_rate
}

with open('enf_results.pkl', 'wb') as f:
    pkl.dump(ENF_save, f)

### Load only
# with open('enf_results.pkl', 'rb') as f:
#     loaded_data = pkl.load(f)

# ENF_drift = loaded_data['drift']
# sampling_rate = loaded_data['fs']

# time_axis = np.linspace(0, len(ENF_drift) / sampling_rate, len(ENF_drift))
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(time_axis, ENF_drift, color='b', label='Extracted ENF')
# ax.axhline(y=100.0, color='r', linestyle='--', alpha=0.7, label='100 Hz Base')
# ax.set_title("ENF Drift")
# ax.set_xlabel("Time")
# ax.set_ylabel("Frequency")
# ax.grid(True)
# ax.legend()
# plt.show()