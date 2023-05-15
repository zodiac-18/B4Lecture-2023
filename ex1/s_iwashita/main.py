import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def stft(data, framesize, overlap):
    """Short-Time Fourier Tranform"""
    win = np.hamming(framesize)
    step = int(framesize * (1 - overlap))
    fft_times = int(data.shape[0] // step) - 1
    spec = []
    for i in range(fft_times):
        start = int(i * step)
        end = start + framesize
        if end > data.shape[0]:
            break
        spec.append(np.fft.fft(data[start:end] + win))
    return np.array(spec)


def istft(spec, framesize, overlap):
    """Inverse STFT"""
    win = np.hamming(framesize)
    step = int(framesize * (1 - overlap))
    istft_times = spec.shape[0] * step + framesize
    data = np.zeros(int(istft_times))
    for i in range(spec.shape[0]):
        start = int(i * step)
        end = start + step
        temp = np.fft.ifft(spec[i, :])
        temp = np.real(temp) / win
        data[start:end] += temp[0:step]

    return data


def main():
    # load a wav file
    filename = "sample.wav"
    data, sr = sf.read(filename)

    # normalize
    data_norm = data / np.abs(data).max()

    # convert horizontal axis to seconds
    x_t = np.arange(0, len(data)) / sr

    # prepare the area to display the graph
    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    # display original signal
    axs[0].plot(x_t, data_norm)
    axs[0].set_title("Original signal")
    axs[0].set_ylabel("Magnitude")
    axs[0].set_xlim(0, len(data) / sr)
    axs[0].set_ylim(-1, 1)

    # calculate the spectrogram
    framesize = 1024
    overlap = 0.9
    spec = stft(data, framesize, overlap)

    # convert to logarithmic scale
    with np.errstate(divide="ignore", invalid="ignore"):
        spec_amp = 20 * np.log10(np.abs(spec[:, : int(framesize // 2 + 1)]))

    # display spectrogram
    axs[1].imshow(
        spec_amp.T,
        aspect="auto",
        origin="lower",
        extent=[0, len(data) / sr, 0, sr / 2000],
        cmap="jet",
    )
    axs[1].set_title("Spectrogram")
    axs[1].set_ylabel("Frequency (kHz)")

    # restore waveform with inverse stft
    re_data = istft(spec, framesize, overlap)
    # normalize
    re_data_norm = re_data / np.abs(re_data).max()

    # convert horizontal axis to seconds
    x_t2 = np.arange(0, len(re_data)) / sr

    sf.write("re-sample.wav", re_data, sr)

    # display re-synthesized signal
    axs[2].plot(x_t2, re_data_norm)
    axs[2].set_title("Re-synthesized signal")
    axs[2].set_ylabel("Magnitude")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_xlim(0, len(data) / sr)
    axs[2].set_ylim(-1, 1)

    plt.savefig("wave.png")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
