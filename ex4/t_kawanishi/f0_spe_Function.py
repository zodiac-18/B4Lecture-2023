"""Fundamental frequency estimation function."""
import numpy as np
import scipy


# Autocorrelation function
def ACF(data: np.ndarray) -> np.ndarray:
    """To adapt autocorrelation function.

    Args:
        data (np.ndarray): As the name implies

    Returns:
        np.ndarray: group of the outcome
    """
    length = len(data)
    result = np.zeros(length)
    data_d = np.concatenate([data, data])

    for i in range(length):
        result[i] = data @ data_d[i: length + i]
    return result


# peak
def peak(result: np.ndarray) -> int:
    """search the peak of the input

    Args:
        result (np.ndarray): input

    Returns:
        int: the peak of the input
    """
    peak = np.zeros(len(result) - 2)
    for i in range(len(result) - 2):
        if result[i] < result[i + 1] and result[i + 1] > result[i + 2]:
            peak[i] = result[i + 1]
    f0 = np.argmax(peak)
    return f0


# estimating f0 with Autocorrelation Function
def f0_ACF(
        data: np.ndarray, sample_rate: int,
        size=100, overlap_r=0.5) -> np.ndarray:
    """To adapt autocorrelation Function for estimate f0.

    Args:
        data (np.ndarray): sound data
        sample_rate (int): sound's sample rate
        size (int, optional): size of frame. Defaults to 100.
        overlap_r (float, optional): overlap rate between
                                    frame and frame. Defaults to 0.5.

    Returns:
        np.ndarray: fundamental group
    """
    # generate var for compute
    dis = int(size * (1 - overlap_r))
    f_num = int(len(data) // (dis))  # amount of the frame
    f0_g = np.zeros(f_num)  # Fundamental frequency group
    win = np.hamming(size)

    # search each frame's f0
    for i in range(f_num):
        if i * dis + size > len(data):
            break
        # ACF return
        result = ACF(data[i * dis: i * dis + size] * win)

        # peak group
        f0 = peak(result)
        if f0 == 0:
            f0_g[i] = 0
        else:
            f0_g[i] = sample_rate / f0
    return f0_g


# cepstrum method
def cepstrum(data: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        data (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    # fft
    fft_data = np.fft.rfft(data)

    # log
    power_spec = np.log10(np.abs(fft_data))

    # ifft
    cep = np.real(np.fft.irfft(power_spec))
    return cep


# estimating f0 with cepstrum method
def f0_cep(
    data: np.ndarray, sample_rate: int, lif=50, size=100, overlap_r=0.5
) -> np.ndarray:
    """To adapt cepstrum method for estimate f0.

    Args:
        data (np.ndarray): sound data
        sample_rate (int): sound's sample rate
        lif (int, optional): the lifter range.Defaults to 50
        size (int, optional): size of frame. Defaults to 100.
        overlap_r (float, optional): overlap rate between frame and frame.
                                     Defaults to 0.5.

    Returns:
        np.ndarray: fundamental group
    """
    # generate var for compute
    dis = int(size * (1 - overlap_r))
    f_num = int(len(data) // (dis))  # amount of the frame
    f0_g = np.zeros(f_num)  # Fundamental frequency group
    win = np.hamming(size)

    # search each frame's f0
    for i in range(f_num):
        if i * dis + size > len(data):
            break

        # adapt to cepstrum
        result = cepstrum(data[i * dis: i * dis + size] * win)

        f0_g[i] = peak(result[lif: (len(result) // 2)])
    return sample_rate / (f0_g + lif)


# levinson durbin algorithm
def levinson(r: np.ndarray, deg: int) -> tuple[np.ndarray, np.ndarray]:
    """Levinson durbin algorithm

    Args:
        r (np.ndarray): autocorrelation data
        deg (int): degree

    Returns:
        tuple[np.ndarray,np.ndarray]: first is coefficient and second is error
    """
    if deg == 1:
        a = np.array([1, -r[1] / r[0]])
        E = a.dot(r[:2])
    else:
        a_s, E_s = levinson(r, deg - 1)
        k_s = -(a_s @ r[deg:0:-1]) / E_s
        U = np.append(a_s, 0)
        V = U[::-1]
        a = U + k_s * V
        E = E_s * (1 - k_s * k_s)

    return a, E


# lpc method
def lpc(data: np.ndarray, deg: int, size=100) -> np.ndarray:
    r = ACF(data)
    a, e = levinson(r[: len(r) // 2], deg)
    w, h = scipy.signal.freqz(np.sqrt(e), a, size, "whole")
    return 20 * np.log10(np.abs(h))


# use cepstrum to estimate spectral envelop
def cep_env(data: np.ndarray, lif=50) -> np.ndarray:
    """To adapt cepstrum method for estimate spectrum envelop.

    Args:
        data (np.ndarray): sound data
        lif (int, optional): the lifter range.Defaults to 50

    Returns:
        np.ndarray: fundamental group
    """
    cep = cepstrum(data)
    cep[lif:-lif] = 0

    return 20 * np.real(np.fft.fft(cep))


if __name__ == "__main__":
    pass
