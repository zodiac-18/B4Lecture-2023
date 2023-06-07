"""Estimate fundamental frequency and spectral envelope."""
import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import soundfile as sf


def loadfile(filename):
    """Load sound file.

    Args:
        filename (str): file name
    Returns:
        data (ndarray): sound data
        sr (int): sample rate
    """
    data, sr = sf.read(filename)
    return data, sr


def autocorrelation(data):
    """Calculate autocorrelation.

    Args:
        data (ndarray): Delimited data

    Returns:
        ndarray: autocorrelation
    """
    length = len(data)
    result = np.zeros(length)
    for i in range(length):
        result[i] = data[: length - i] @ data[i:]
    return result


def f0_estimate_autocorrelation(data, sr, framesize=512, overlap=256):
    """Estimate the fundamental frequency using the autocorrelation method.

    Args:
        data (ndarray): Sound data
        sr (int): Sample rate
        framesize (int, optional): Size of frame. Defaults to 512.
        overlap (int, optional): Size of overlap. Defaults to 256.

    Returns:
        ndarray: Estimated fundamental frequency
    """
    steps = (len(data) - overlap) // (framesize - overlap)
    autocorrelation_ = np.zeros((framesize, steps))
    peaks = np.zeros(steps)
    for i in range(steps):
        frame = data[i * (framesize - overlap) : i * (framesize - overlap) + framesize]

        autocorrelation_[:, i] = autocorrelation(frame)

        maximul = np.zeros((0, 2))
        for j in range(framesize - 2):
            if (
                autocorrelation_[j, i] < autocorrelation_[j + 1, i]
                and autocorrelation_[j + 1, i] > autocorrelation_[j + 2, i]
            ):
                maximul = np.append(
                    maximul, np.array([[(j + 1), autocorrelation_[j + 1, i]]]), axis=0
                )
        peaks[i] = maximul[np.argmax(maximul[:, 1]), 0]

    return sr / peaks


def cepstrum(data):
    """Calculate cepstrum.

    Args:
        data (ndarray): Input data

    Returns:
        ndarray: Cepstrum
    """
    length = len(data)
    window = np.hamming(length)
    windowed_data = data * window
    spec = np.fft.fft(windowed_data)
    spec_db = 20 * np.log10(np.abs(spec))

    return np.real(np.fft.ifft(spec_db))


def short_time_cepstrum(data, framesize: int = 512, overlap: int = 256):
    """Calculate short time cepstrum.

    Args:
        data (ndarray): Input data
        framesize (int, optional): Size of frame. Defaults to 512.
        overlap (int, optional): Size of overlap. Defaults to 256.

    Returns:
        ndarray: Short time cepstrum
    """
    steps = (len(data) - overlap) // (framesize - overlap)
    cepstrums = np.zeros((steps, framesize))
    for i in range(steps):
        cepstrums[i] = cepstrum(
            data[i * (framesize - overlap) : i * (framesize - overlap) + framesize]
        )

    return cepstrums


def generate_lifter(length, cutoff=50, mode="hp"):
    """Genarate lifter.

    Args:
        length (int): Length of lifter
        cutoff (int, optional): Cutoff cepstrum coefficient. Defaults to 50.
        mode (str, optional): "hp" of "lp". Defaults to "hp".

    Returns:
        ndarray: Lifter
    """
    lifter = np.zeros(length)
    lifter[cutoff : length - cutoff] = 1
    if mode == "lp":
        lifter = 1 - lifter

    return lifter


def f0_estimate_cepstrum(data, sr, framesize: int = 512, overlap: int = 256):
    """Estimate the fundamental frequency using the cepstrum method.

    Args:
        data (ndarray): Sound data
        sr (int): Sample rate
        framesize (int, optional): Size of frame. Defaults to 512.
        overlap (int, optional): Size of overlap. Defaults to 256.

    Returns:
        ndarray: Estimated fundamental frequency
    """
    cepstrums = short_time_cepstrum(data)

    liftered_cepstrums = cepstrums * generate_lifter(framesize, mode="hp")

    peaks = np.argmax(liftered_cepstrums[:, : framesize // 2], axis=1)

    return sr / peaks


def log_spectrum(data):
    """Calculate log amplitude spectrum.

    Args:
        data (ndarray): Input data

    Returns:
        ndarray: log amplitude spectrum
    """
    spectrum = np.fft.fft(data * np.hamming(len(data)))

    return 20 * np.log10(np.abs((spectrum)))


def envelope_cepstrum(data):
    """Calculate spectrum envelope by cepstrum.

    Args:
        data (ndarray): Input data

    Returns:
        ndarray: Spectrum envelope by cepstrum
    """
    cepstrum_ = cepstrum(data)
    lifter = generate_lifter(len(data), mode="lp")

    return np.real(np.fft.fft(cepstrum_ * lifter))


def envelope_lpc(data, p, sr):
    """Calculate spectrum envelope by LPC.

    Args:
        data (ndarray): Input data
        p (int): dimention of LPC
        sr (int): sample rate

    Returns:
        nadrray: Spectrum envelope by LPC
    """
    window = np.hamming(len(data))
    windowed_data = data * window
    autocorrelation_ = autocorrelation(windowed_data)
    alphas, e = levinson_durbin(autocorrelation_, p)
    w, h = scipy.signal.freqz(np.sqrt(e), alphas, whole=True, fs=sr)

    return 20 * np.log10(np.abs(h))


def levinson_durbin(data, dimension):
    """Calculate linear prediction coefficients.

    Args:
        data (ndarray): Autocorrelation
        dimension (int): Dimention of LPC

    Returns:
        ndarray: Linear prediction coefficients
    """
    result = np.zeros(dimension + 1)

    result[0] = 1
    result[1] = -data[1] / data[0]
    residual_error = data[0] + result[1] * data[1]

    for i in range(1, dimension):
        lambda_ = -np.sum(result[0 : i + 2] * np.flip(data[: i + 2])) / residual_error

        result[: i + 2] += lambda_ * np.flip(result[: i + 2])
        residual_error *= 1 - np.power(lambda_, 2)

    return result, residual_error


def main():
    """Plot f0 and the cepstrum envelope."""
    parser = argparse.ArgumentParser(
        description="""Estimation of fundamental frequency
        and calculation of spectral envelope"""
    )

    parser.add_argument("name", help="file name")

    args = parser.parse_args()

    filename = args.name
    data, sr = loadfile(filename)

    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4))
    fig1.subplots_adjust(wspace=0.4)

    data_stft = librosa.stft(data)
    spectrogram, phase = librosa.magphase(data_stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)

    f0_estimated_autocorrelation = f0_estimate_autocorrelation(data, sr)
    f0_estimated_cepstrum = f0_estimate_cepstrum(data, sr)
    t = np.linspace(0, len(data) / sr, len(f0_estimated_autocorrelation))

    ax1[0].set_title("f0 by autocorrelation")
    ax1[0].plot(t, f0_estimated_autocorrelation, label="f0", color="c")
    im1 = librosa.display.specshow(
        spectrogram_db, sr=sr, ax=ax1[0], x_axis="time", y_axis="log", cmap="plasma"
    )
    ax1[0].set_xlabel("Time [s]")
    ax1[0].set_ylabel("Frequency [Hz]")
    ax1[0].set_xlim(0, t[-1])
    ax1[0].set_xticks(np.arange(0, t[-1], 1))
    ax1[0].legend()
    fig1.colorbar(im1, ax=ax1[0], format="%+2.f dB")

    ax1[1].set_title("f0 by cepstrum")
    ax1[1].plot(t, f0_estimated_cepstrum, label="f0", color="c")
    im2 = librosa.display.specshow(
        spectrogram_db, sr=sr, ax=ax1[1], x_axis="time", y_axis="log", cmap="plasma"
    )
    ax1[1].set_ylabel("Frequency [Hz]")
    ax1[1].set_xlabel("Time [s]")
    ax1[1].set_xlim(0, len(data) / sr)
    ax1[1].set_xticks(np.arange(0, t[-1], 1))
    ax1[1].legend()
    fig1.colorbar(im2, ax=ax1[1], format="%+2.f dB")

    # plt.show()
    plt.savefig("plot_f0.png")

    fig2, ax2 = plt.subplots()

    frame_size = 512
    target_frame_start = int(len(data) * 0.2)
    target_frame = data[target_frame_start : target_frame_start + frame_size]
    log_spectrum_ = log_spectrum(target_frame)
    f = np.linspace(0, sr // 2, frame_size // 2)

    envelope_cepstrum_ = envelope_cepstrum(target_frame)

    p = 32
    envelope_lpc_ = envelope_lpc(target_frame, p, sr)

    ax2.plot(f, log_spectrum_[: frame_size // 2], label="Spectrum")
    ax2.plot(f, envelope_cepstrum_[: frame_size // 2], label="Cepstrum")
    ax2.plot(f, envelope_lpc_[: frame_size // 2], label="LPC")
    ax2.set_ylabel("Log amplitude spectrum [dB]")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.legend()

    # plt.show()
    plt.savefig("envelope.png")


if __name__ == "__main__":
    main()
