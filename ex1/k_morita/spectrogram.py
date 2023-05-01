"""Spectral processing modules."""
import numpy as np


def stft(y, sr, win_func=np.hamming, win_len=2048, ol=0.75):
    """Short-time Fourier Transform.

    Args:
        y (np.ndarray): input signal
        sr (int): sampling rate
        win_func (function, optional): window function. Defaults to np.hamming.
        win_len (int, optional): window length. Defaults to 2048.
        ol (float, optional): overlap rate. Defaults to 0.5.

    Returns:
        tuple(np.ndarray, float, int):
            fft_array(np.ndarray): complex-valued stft matrix
            total_time(float): total signal time [s]
            total_frame(int): total signal frame
    """
    window = win_func(win_len)  # create window
    hop_len = int(win_len * (1 - ol))  # length of a hop
    n_steps = (len(y) - win_len) // hop_len  # number of fft steps
    total_frame = hop_len * n_steps + win_len  # number of total frame
    total_time = total_frame / sr  # total time [s]

    # fft
    fft_array = [
        np.fft.fft(window * y[i * hop_len : i * hop_len + win_len])
        for i in range(n_steps)
    ]
    # convert list -> np.ndarray
    fft_array = np.array(fft_array).T

    # remove frequency element(from nyquist to samplerate)
    fft_array = fft_array[fft_array.shape[0] // 2 :]

    return fft_array, total_time, total_frame


def istft(fft_array, sr, win_func=np.hamming, win_len=2048, ol=0.75):
    """Inverse SHort-time Fourier Transform.

    Args:
        fft_array (np.ndarray): stft matrix
        sr (int): sampling rate
        win_func (function, optional): window function. Defaults to np.hamming.
        win_len (int, optional): window length. Defaults to 2048.
        ol (float, optional): overlap rate. Defaults to 0.75.

    Returns:
        tuple(np.ndarray, float, int):
            data(np.ndarray): complex-valued matrix reconstructed from stft
            total_time(float): total signal time [s]
            total_frame(int): total signal frame
    """
    # recover fft_array
    if win_len & 1:  # if window length is odd
        fft_array = np.concatenate([fft_array, fft_array[::-1, :][:-1]])
    else:
        fft_array = np.concatenate([fft_array, fft_array[::-1, :][:]])

    fft_array = fft_array.T

    window = win_func(win_len)  # create window
    hop_len = int(win_len * (1 - ol))  # length of a hop
    n_steps = len(fft_array)  # number of fft steps
    total_frame = hop_len * n_steps + win_len  # number of total frame
    total_time = total_frame / sr  # total time [s]

    # ifft
    ifft_array = [np.fft.ifft(fft) / window for fft in fft_array]

    # covert list -> np.ndarray
    ifft_array = np.array(ifft_array)

    # craete wave
    data = np.zeros(total_frame, dtype=np.float32)
    for i, ifft in enumerate(ifft_array):
        left = i * hop_len
        right = left + win_len
        data[left:right] = np.real(ifft)

    return data, total_time, total_frame


def magphase(complex):
    """Calculate magnitude and phase from complex-valued stft matrix.

    Args:
        complex (np.ndarray): complex-valued stft matrix

    Returns:
        tuple (np.ndarray, np.ndarray):
            mag (np.ndarray): magnitude
            phase (np.ndarray): phase
    """
    mag = np.abs(complex)
    phase = np.exp(1.0j * np.angle(complex))
    return mag, phase


def mag_to_db(mag):
    """Convert an mag spectrogram into dB-scaled spectrogram.

    Args:
        mag (np.ndarray): magnitude

    Returns:
        db (np.ndarray): dB-scaled magnitude
    """
    db = 20 * np.log10(mag)
    return db
