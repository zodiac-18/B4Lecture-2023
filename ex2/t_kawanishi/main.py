"""To generate spectrogram."""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import ex1Function as F
import filters as flt

parser = argparse.ArgumentParser(
    description="This is a program to generate sound's spectrogram"
)
parser.add_argument("path", help="the path of sound")
parser.add_argument(
    "-s",
    "--f_size",
    help="frame size recommend between 128 to 2048 ",
    default=512,
    type=int,
)
parser.add_argument(
    "-r", "--overlap_r", help="overlap rate between 0 to 1", default=0.5, type=float
)
parser.add_argument(
    "-l", "--y_limit", help="limit of spectrogram's frequency", default=25000, type=int
)


# convolution
def conv(data: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """To adapt convolution with data an h.

    Args:
        data (np.ndarray): signal which needed convolution
        filter (np.ndarray): signal for convolution

    Returns:
        np.ndarray: the outcome of convolution with data and h
    """
    y = np.zeros(len(data) + len(filter) - 1)
    for i in range(len(data)):
        y[i : i + len(filter)] += data[i] * filter
    return y[: len(data)]


if __name__ == "__main__":
    # get parser
    args = parser.parse_args()
    Fs = args.f_size
    overlap_r = args.overlap_r
    spec_y_lim = args.y_limit

    # get sound
    data, sample_rate = F.load_sound(args.path)

    # create a filter and adapt to sound
    filter = flt.bpf(500, 10000, sample_rate, Fs)
    filtered_data = conv(data, filter)

    # compute some value for plot
    x_t = np.arange(0, len(data)) / sample_rate  # sound's time array

    # adapt fourier transform to filter
    filter_i = np.fft.fft(filter)
    frame = filter_i[: len(filter_i) // 2]

    # get corresponding frequency array
    freq = np.fft.fftfreq(len(filter), d=1 / sample_rate)
    freq = freq[: len(filter) // 2]

    # get angle
    angle = np.angle(frame)

    # plot graph
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(freq, 20 * np.log(np.abs(frame)))
    axs[0, 0].set_xlabel("Frequency[Hz]")
    axs[0, 0].set_ylabel("Amplitude[dB]")
    axs[0, 0].set_title("BPF's Frequency Response(Amplitude)")

    axs[0, 1].plot(np.arange(0, Fs + 1), filter)
    axs[0, 1].set_xlabel("Sample")
    axs[0, 1].set_ylabel("Sound pressure[Pa]")
    axs[0, 1].set_title("filter preview")

    axs[1, 1].plot(freq, angle)
    axs[1, 1].set_xlabel("Frequency[Hz]")
    axs[1, 1].set_ylabel("Phase[rad]")
    axs[1, 1].set_title("BPF's Frequency Response(Phase)")

    axs[1, 0].set_xlabel("Time[s]")
    axs[1, 0].set_ylabel("Sound pressure[Pa]")
    axs[1, 0].set_title("sound's difference")
    axs[1, 0].plot(x_t, data, label="original sound")
    axs[1, 0].plot(x_t, filtered_data, label="filtered sound")
    axs[1, 0].legend(loc=0)  # 凡例
    plt.savefig("every_graph.png")

    F.show_spectrogram(
        data, overlap_r=overlap_r, Fs=Fs, y_lim=spec_y_lim, s_name="origin"
    )
    F.show_spectrogram(
        filtered_data, overlap_r=overlap_r, Fs=Fs, y_lim=spec_y_lim, s_name="filtered"
    )
    F.extract_sound(filtered_data, file_name="Re_voice", sample_rate=48000)
    plt.show()
