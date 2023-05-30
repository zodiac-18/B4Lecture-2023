"""This file has functions to read wavfiles and handle spectrograms."""
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable


def wavload(path):
    """
    Read audio file.

    Args:
        path(str): Path of audio file.

    Returns:
        ndarray: A list of audio data
        float: A samplerate of the audio file
    """
    data, samplerate = sf.read(path)
    return data, samplerate


def STFT(data, WIDTH):
    """Read audio data and do STFT.

    Args:
        data (ndarray): audio data to do STFT.
        WIDTH (int): Window width of STFT.

    Returns:
        ndarray(complex): Result of STFT in complex
    """
    OVERLAP = int(WIDTH / 2)
    # Number of audio segments
    split_number = len(np.arange((WIDTH / 2),
                                 data.shape[0], (WIDTH - OVERLAP)))
    # Size of Fourier transformed data with splited
    fframe_size = len(np.fft.fft(data[:WIDTH]))

    spec = np.zeros([split_number, fframe_size], dtype=complex)
    window = np.hamming(WIDTH)
    pos = 0  # Position of audio segment

    # STFT
    for i in range(split_number):
        frame = data[pos: pos + WIDTH]
        if len(frame) >= WIDTH:
            windowed = window * frame
            # Fourier transform of segmented audio
            fft_result = np.fft.fft(windowed)
            spec[i] = fft_result
            pos += OVERLAP
    return spec


def spectrogram(TOTAL_TIME, samplerate, data):
    """Show Spectrogram.

    Args:
        TOTAL_TIME (float): Audio file's total time.
        samplerate (float): A samplerate of audio file.
        data (ndarray): Result of STFT in complex.

    Returns:
        None
    """
    freq = np.linspace(0, samplerate / 2, data.shape[1] // 2)
    amp = np.abs(data[:, data.shape[1] // 2 - 1:: -1])
    amp = 20 * np.log10(amp)

    plt.rcParams["image.cmap"] = "jet"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.imshow(amp.T[freq<=8000, :], extent=[0, TOTAL_TIME, 0, 8000], aspect="auto")
    plt.colorbar()
    plt.xlim(0, TOTAL_TIME)
    plt.ylim(0, 8000)
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.show()


def spectrogram_double(TOTAL_TIME1, TOTAL_TIME2, samplerate, data1, data2):
    """Show Spectrogram with subplot (Original and Filtered).

    Args:
        TOTAL_TIME1 (float): A total time of data1.
        TOTAL_TIME2 (float): A total time of data2.
        samplerate (float): A samplerate of two data
        data1 (_type_): Original Signal of the sound.
        data2 (_type_): Filtered Signal of the sound.

    Returns:
        None
    """
    plt.rcParams["image.cmap"] = "jet"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    amp1 = np.abs(data1[:, int(data1.shape[1] / 2):: -1])
    amp1 = np.log(amp1**2)
    amp2 = np.abs(data2[:, int(data2.shape[1] / 2):: -1])
    amp2 = np.log(amp2**2)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(211)
    ax1.set_title("Spectrogram")
    ax1.set_title("Original")
    im = ax1.imshow(
        amp1.T,
        extent=[0, TOTAL_TIME1, 0, samplerate / 2],
        aspect="auto",
        vmax=10,
        vmin=-25,
    )
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im, cax=cax1)
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlim(0, TOTAL_TIME1)
    ax1.set_ylim(0, samplerate / 2)

    ax2 = fig.add_subplot(212)
    ax2.set_title("Filtered")
    im = ax2.imshow(
        amp2.T,
        extent=[0, TOTAL_TIME2, 0, samplerate / 2],
        aspect="auto",
        vmax=10,
        vmin=-25,
    )
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im, cax=cax2)
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlim(0, TOTAL_TIME2)
    ax2.set_ylim(0, samplerate / 2)

    fig.tight_layout()
    plt.show()


def convolution(x, h):
    """Receives arrays x and h, and returns result fo the convolution y."""
    y = np.zeros(len(x) + len(h) - 1)
    width = len(h)
    for i in range(len(x)):
        y[i: i + width] += x[i] * h

    return y


def LPF_window(length, cutoff, size, samplerate):
    """
    Create a filter and Returns FIR impulse respoonse.

    Args:
        length (int):   Order of the filter. it must be odd number.
        cutoff (float): Cutoff frequency of the filter.
        size (int):     Size of filter.
        samplerate(int):samplerate of filtered signal.

    Returns:
        ndarray[int]:   Impulse response of the FIR filter
    """
    N = int((length - 1) / 2)  # tap number
    nyq_sample = int(size / 2)  # nyquist frequency [sample]
    window = np.kaiser(length, 6)  # kaiser window
    freq = np.linspace(0, samplerate, size)
    F_filter = np.zeros(size)

    # Filter Design
    F_filter[freq < cutoff] = 1
    im_response = np.fft.ifft(F_filter)
    im_response[N: -N] = 0  # Censoring of impulse response
    im_response = np.fft.ifftshift(im_response)
    # multiplying window function
    windowed = window * im_response[nyq_sample - N: nyq_sample + N + 1]
    im_response[nyq_sample - N: nyq_sample + N + 1] = windowed
    im_response = np.fft.fftshift(im_response)
    F_filter = np.fft.fft(im_response)
    h = np.roll(im_response, N)  # FIR filter's impulse response
    filter = np.fft.fft(h)

    # Show Filter Properties
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12
    plt.plot(window)
    plt.show()
    plt.subplot(211)
    plt.title("FIR properties")
    plt.plot(freq, 20 * np.log(np.abs(filter)))
    plt.ylabel("Amplitude [dB]")
    plt.subplot(212)
    plt.plot(freq, np.arctan2(np.imag(filter), np.real(filter)))
    plt.ylabel("Phase [rad]")
    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()

    return np.real(h[:length])
