"MFCC."
import numpy as np
from scipy.fft import dct


def melfilterbank(nfilt, NFFT, pow_frames, bin):
    """Calculate Melfilter Bank.
    Args:
        nfilt (int): nubmer of filter
        NFFT (int): number of nfft points
        pow_frames (ndarray): arra of power frame
        bin (ndarray): array
    Returns:
        ndarray: apply mel filter bank
    """

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(
        filter_banks == 0, np.finfo(float).eps, filter_banks
    )  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return filter_banks


def DCT(filter_banks, num_ceps):
    """Calculate the Discrete Cosine Transform.
    Args:
        filter_banks (ndarray): array of filter bank
        num_ceps (int): number of cepstrum
    Returns:
        ndarray: array of mfcc
    """
    mfcc = dct(filter_banks, type=2, axis=1, norm="ortho")[
        :, 1: (num_ceps + 1)
    ]
    mfcc -= np.mean(mfcc, axis=0) + 1e-8

    return mfcc


def delta(mfcc):
    """Calculate dynamic variable component.
    Args:
        mfcc (ndarray): array of mfcc
    Returns:
        ndarray: dynamic variable component
    """
    p_mfcc = np.append(mfcc, np.zeros((1, len(mfcc[0]))), axis=0)

    delta = np.zeros(mfcc.shape)
    for i in range(1, len(p_mfcc) - 1):
        delta[i] = p_mfcc[i + 1] - p_mfcc[i - 1]
    return delta


def mfcc(data, fs, nfilt, NFFT, ceps):
    """Calculate mfcc.
    Args:
        data (ndarray): target data
        fs (int): sampling rate
        nfilt (int): number of filter
        NFFT (int): number of fft points
        ceps (int): number of cepstrum
    Returns:
        ndarray: data of mfcc
    """
    frame_size = 0.025
    frame_len = 0.01
    pre_emphasis = 0.97

    emph_signal = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
    frame_length, frame_step = (
        frame_size * fs,
        frame_len * fs,
    )  # Convert from seconds to samples
    signal_length = len(emph_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)
    )  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(
        emph_signal, z
    )

    indices = (
        np.tile(np.arange(0, frame_length), (num_frames, 1))
        + np.tile(
            np.arange(0, num_frames * frame_step, frame_step),
            (frame_length, 1)
        ).T
    )
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(
        low_freq_mel, high_freq_mel, nfilt + 2
    )  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / fs)

    filter_banks = melfilterbank(nfilt, NFFT, pow_frames, bin)
    mfcc = DCT(filter_banks, ceps)

    return mfcc