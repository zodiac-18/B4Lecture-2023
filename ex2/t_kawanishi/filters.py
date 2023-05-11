"""Some function to create filter."""
import numpy as np


# bond pass filter
def bpf(low_f: int, high_f: int, sample_rate: int, f_size: int) -> np.ndarray:
    """Generate bond pass filter.

    Args:
        low_f (int): low point of bond
        high_f (int): high point of bond
        sample_rate (int): as the name implies
        f_size (int): the size of filter

    Returns:
        np.ndarray: filter
    """
    # changing frequency to angular frequency
    low_w = 2 * np.pi * low_f / sample_rate
    high_w = 2 * np.pi * high_f / sample_rate

    # create filter
    filter = np.zeros(f_size + 1)
    cnt = 0
    for i in range(-f_size // 2, f_size // 2 + 1):
        filter[cnt] = (
            high_w * np.sinc(high_w * i / np.pi) - low_w * np.sinc(low_w * i / np.pi)
        ) / np.pi
        cnt += 1

    return filter * np.hamming(f_size + 1)


# bond elimination filter
def bef(low_f: int, high_f: int, sample_rate: int, f_size: int) -> np.ndarray:
    """Generate bond elimination filter.

    Args:
        low_f (int): low point of bond
        high_f (int): high point of bond
        sample_rate (int): as the name implies
        f_size (int): filter's size

    Returns:
        np.ndarray: filter
    """
    # changing frequency to angular frequency
    low_w = 2 * np.pi * low_f / sample_rate
    high_w = 2 * np.pi * high_f / sample_rate
    # create filter
    filter = np.zeros(f_size + 1)
    cnt = 0
    for i in range(-f_size // 2, f_size // 2 + 1):
        filter[cnt] = (
            np.sinc(i)
            + (
                -high_w * np.sinc(high_w * i / np.pi)
                + low_w * np.sinc(low_w * i / np.pi)
            )
            / np.pi
        )
        cnt += 1

    return filter * np.hamming(f_size + 1)


# low pass filter
def lpf(frequency: int, sample_rate: int, f_size: int) -> np.ndarray:
    """Generate low pass filter.

    Args:
        frequency (int): cutoff frequency
        sample_rate (int): as the name implies
        f_size (int): filter's size

    Returns:
        np.ndarray: filter
    """
    # changing frequency to angular frequency
    w_f = 2 * np.pi * frequency / sample_rate

    # create filter
    filter = np.zeros(f_size + 1)
    cnt = 0
    for i in range(-f_size // 2, f_size // 2 + 1):
        filter[cnt] = w_f * np.sinc(w_f * i / np.pi) / np.pi
        cnt += 1

    return filter * np.hamming(f_size + 1)


# high pass filter
def hpf(frequency: int, sample_rate: int, f_size: int) -> np.ndarray:
    """Generate high pass filter.

    Args:
        frequency (int): cutoff frequency
        sample_rate (int): as the name implies
        f_size (int): filter size

    Returns:
        np.ndarray: filter
    """
    # changing frequency to angular frequency
    w_f = 2 * np.pi * frequency / sample_rate
    # create filter
    filter = np.zeros(f_size + 1)
    cnt = 0
    for i in range(-f_size // 2, f_size // 2 + 1):
        filter[cnt] = (
            np.sinc(np.pi * i / np.pi) - (w_f * np.sinc(w_f * i / np.pi)) / np.pi
        )
        cnt += 1

    return filter * np.hamming(f_size + 1)
