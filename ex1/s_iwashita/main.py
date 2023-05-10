import numpy as np
import matplotlib.pyplot as plt
import librosa


def stft(x, fft_size, hop_size):
    """Short-Time Fourier Tranform"""
    win = np.hamming(fft_size)
    half_fft_size = fft_size // 2 + 1
    spec = np.empty((half_fft_size, len(x) // hop_size), dtype=np.complex128)
    for i in range(spec.shape[1]):
        start = i * hop_size
        end = start + fft_size
        if end > len(x):
            break
        spec[:, i] = np.fft.rfft(win * x[start:end], n=fft_size)
    return spec


def istft(spec, fft_size, hop_size):
    """Inverse STFT"""
    win = np.hamming(fft_size)
    x = np.zeros((len(spec[0]) - 1) * hop_size + fft_size)
    for i in range(spec.shape[1]):
        start = i * hop_size
        end = start + fft_size
        if end > len(x):
            break
        x[start:end] += win * np.fft.irfft(spec[:, i], n=fft_size)
    return x

# load a wav file
filename = 'C:\\Users\\iwass\\Music\\iTunes\\iTunes Media\\Music\\Unknown Artist\\Unknown Album\\sample.wav'
data, sr = librosa.load(filename, sr=None)

# normalize
data_norm = data / np.abs(data).max()

# convert horizontal axis to seconds
x_t = np.arange(0, len(data)) / sr

# prepare the area to display the graph
fig, axs = plt.subplots(3, 1, figsize=(6, 6))

# display original signal
axs[0].plot(x_t, data_norm)
axs[0].set_title('Original signal')
axs[0].set_ylabel('Magnitude')
axs[0].set_xlim(0, len(data)/sr)
axs[0].set_ylim(-1, 1)

# calculate the spectrogram
n_fft = 256
hop_length = n_fft // 4
S = np.abs(stft(data_norm, fft_size=n_fft, hop_size=hop_length))**2

# convert to logarithmic scale
S_db = librosa.amplitude_to_db(S, ref=np.max)

# display spectrogram
axs[1].imshow(S_db, aspect='auto', origin='lower', cmap='jet', extent=[0, data_norm.shape[0]/sr, 0, sr/2/1000])
axs[1].set_title('Spectrogram')
axs[1].set_ylabel('Frequency (kHz)')

# restore waveform with inverse stft
y_inv = istft(S, fft_size=n_fft, hop_size=hop_length)

# normalize
y_inv_norm = y_inv / np.abs(y_inv).max()

# convert horizontal axis to seconds
y_t = np.arange(0, len(y_inv)) / sr

# display re-synthesized signal
axs[2].plot(y_t, y_inv_norm)
axs[2].set_title('Re-synthesized signal')
axs[2].set_ylabel('Magnitude')
axs[2].set_xlabel('Time (s)')
axs[2].set_xlim(0, len(data)/sr)
axs[2].set_ylim(-1, 1)

plt.tight_layout()
plt.show()
