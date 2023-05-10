"""Filtering methods."""

import numpy as np


def create_lpf(fc, sr, N, win_func=np.hamming):
    """Create LPF function.

    Args:
        fc (int): cut-off freqency
        sr (int): sample rate
        N (int): tap num
        win_func (function, optional): window function. Defaults to np.hamming.

    Returns:
        lpf (np.ndarray): low pass filter matrix
    """
    fc_norm = fc / (sr // 2)
    ideal_lpf = np.sinc(fc_norm * (np.arange(N) - (N - 1) / 2))
    window = win_func(N)

    return ideal_lpf * window


def conv(input, filter):
    """Convolve function.

    Args:
        input (np.ndarray): input matrix
        filter (np.ndarray): filter matrix

    Returns:
        output (np.ndarray): convolved matrix
    """
    filter_len = len(filter)

    extra_arr = np.zeros(len(filter) - 1)
    input = np.concatenate([extra_arr, input, extra_arr])

    output_len = len(input) - len(extra_arr)
    output = np.zeros(output_len)

    filter = filter[::-1].T
    for i in range(output_len):
        left, right = i, i + filter_len
        output[i] = np.dot(input[left:right], filter)

    return output
