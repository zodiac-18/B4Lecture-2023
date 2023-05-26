"""To generate spectrogram."""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

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


def load_sound(sound_path: str) -> tuple[np.ndarray, int]:
    """Load sound file.

    Args:
        sound_path (str): the path of the input sound file

    Returns:
        tuple[np.ndarray, int]: sound's signal value, sample rate
    """
    data, sample_rate = sf.read(file=sound_path)
    return data, sample_rate


def stft(
    data: np.ndarray, overlap: float, Fs: int, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """short-time fourier transform.

    Args:
        data (np.ndarray): sound's signal value
        overlap (float): overlap rate
        Fs (int): frame size
        sample_rate (int): sample rate

    Returns:
        tuple[np.ndarray, np.ndarray, int]: first one is the group of the outcome of data
                                            adapted by short-time fourier transform.
                                            second one is each data's corresponding frequency.
                                            last one is last frame's start point
    """
    frame_dist = int(Fs * (1 - overlap))  # distance between segment and segment
    frame_group = []  # list to collect frame
    win = np.hamming(Fs)  # create windows

    # loop to cut data into segment and adapt hamming window & fft
    frame_s = 0
    while True:
        if frame_s + Fs > data.shape[0]:
            break
        frame = np.fft.fft(data[frame_s : frame_s + Fs] * win)
        frame_group.append(frame)
        frame_s += frame_dist
    freq = np.fft.fftfreq(Fs, d=1 / sample_rate)
    frame_positive = np.delete(
        frame_group, slice(int(Fs / 2) - 1, Fs), 1
    )  # cut negative data
    frame_positive = frame_positive.T
    return frame_positive, freq[1 : int(Fs / 2)], frame_s


def istft(data: np.ndarray, overlap: float, length: int) -> np.ndarray:
    """Inverse short-time fourier transform.

    Args:
        data (np.ndarray): spectrogram's data matrix
        overlap (float): overlap rate
        length (int): the sound's signal data size

    Returns:
        np.ndarray: the sound's signal data
    """
    # create a list to generate sound data
    origin_sound = np.zeros(length)
    seg_s = 0
    Fs = len(data[0])
    size = int(Fs * (1 - overlap))
    win = np.hamming(Fs)  # create windows
    # using ifft to inverse segment
    for i in range(len(data)):
        origin_sound[seg_s : seg_s + size] += (np.real(np.fft.ifft(data[i])) / win)[
            0:size
        ]
        seg_s += int(Fs * (1 - overlap))
    return origin_sound


def show_spectrogram(
    data: np.ndarray, overlap_r=0.5, Fs=512, sample_rate=48000, y_lim=20000, s_name=""
) -> None:
    """To show spectrogram and save.

    Args:
        data (np.ndarray): sound signal
        overlap_r (float, optional): overlap_rate. Defaults to 0.5.
        Fs (int, optional): frame size. Defaults to 512.
        sample_rate (int, optional): as the name implies. Defaults to 48000.
        y_lim (int, optional): spectrogram's y-axis range. Defaults to 20000.
        s_name (str, optional): if want save the image, type a name to save
                                Defaults will not save the image. Defaults to "".
    """
    # adapt data to short-time fourier transform
    frame_group, freq, frame_l = stft(data, overlap_r, Fs, sample_rate)

    # compute time
    spec_t = (
        np.arange(start=0, stop=frame_l, step=int(Fs * (1 - overlap_r))) / sample_rate
    )

    # plot spectrogram
    group = np.zeros(((len(frame_group)), len(frame_group[0])))
    for i in range(len(frame_group)):
        for j in range(len(frame_group[i])):
            x = np.abs(frame_group[i][j])
            if x == 0:
                group[i][j] = x
            else:
                group[i][j] = 20 * np.log(x)
    fig, ax = plt.subplots(figsize=(5, 5))
    pcm = ax.pcolormesh(
        spec_t,
        freq,
        group,
        cmap="plasma",
        shading="nearest",
        vmin=-500,
        vmax=50,
    )

    # add colorbar
    color_b = plt.colorbar(pcm, ax=ax)

    # set y limit
    if y_lim <= max(freq):  # if maximum freq bigger than limit then set y-axis limit
        ax.set_ylim(0, y_lim)
    ax.set_xlabel("Time[s]")
    ax.set_ylabel("Frequency[Hz]")
    ax.set_title("Spectrogram")
    color_b.set_label("Amplitude[dB]", labelpad=-0.1)
    if not s_name == "":
        fig.savefig(s_name)
    return None


def extract_sound(data: np.ndarray, file_name: str, sample_rate: int) -> None:
    """To write sound signal to a wav file.

    Args:
        data (np.ndarray): sound signal
        file_name (str): the name tha wav file to be
        sample_rate (int): as the name implies
    """
    sf.write(file=file_name + ".wav", data=data, samplerate=sample_rate)
    return None


if __name__ == "__main__":
    args = parser.parse_args()
    Fs = args.f_size
    overlap_r = args.overlap_r
    spec_y_lim = args.y_limit
    data, sample_rate = load_sound(args.path)
    frame_group, freq, frame_l = stft(data, overlap_r, Fs, sample_rate)
    origin_sound = istft(frame_group, overlap_r, data.shape[0])

    # compute time
    x_t = np.arange(0, len(data)) / sample_rate
    spec_t = (
        np.arange(start=0, stop=frame_l, step=int(Fs * (1 - overlap_r))) / sample_rate
    )

    # create graph's group
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # create subplot domain
    base_sound = axs[0]
    sound_spec = axs[1]
    repro_sound = axs[2]
    fig.subplots_adjust(hspace=0.5)

    # set x limit
    base_sound.set_xlim(0, len(data) / sample_rate)
    repro_sound.set_xlim(0, len(data) / sample_rate)

    # set y limit
    if spec_y_lim <= max(
        freq
    ):  # if maximum freq bigger than limit then set y-axis limit
        sound_spec.set_ylim(0, spec_y_lim)

    # plot data
    base_sound.plot(x_t, data)
    cax = fig.add_axes([0.92, 0.395, 0.02, 0.2])  # x, y, width, height
    spec_d = sound_spec.pcolormesh(
        spec_t,
        freq[1 : int(Fs / 2)],
        20 * np.log(np.abs(frame_group)),
        cmap="plasma",
        shading="nearest",
    )
    color_b = fig.colorbar(spec_d, cax=cax)
    color_b.set_label("Amplitude[dB]", labelpad=-0.1)
    repro_sound.plot(x_t, origin_sound)

    # set x label
    base_sound.set_xlabel("Time[s]")
    sound_spec.set_xlabel("Time[s]")
    repro_sound.set_xlabel("Time[s]")

    # set y label
    base_sound.set_ylabel("Sound pressure[Pa]")
    sound_spec.set_ylabel("Frequency[Hz]")
    repro_sound.set_ylabel("Sound pressure[Pa]")

    # set title
    base_sound.set_title("Original signal")
    sound_spec.set_title("Spectrogram")
    repro_sound.set_title("Re-synthesized signal")

    plt.show()
    fig.savefig("comparison_graph_ex1.png")
    sf.write(file="Re_voice.wav", data=origin_sound, samplerate=sample_rate)
