"""ex2 high pass filtering."""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile


def conv(left: np.ndarray, right: np.ndarray):
    """Calculate convolution.

    Args:
        left (np.ndarray): The left argument.
        right (np.ndarray): The right argument.

    Returns:
        ndarray: The result of convolution.
    """
    conv_result = np.zeros(len(left) + len(right) - 1)
    for i in range(len(left)):
        conv_result[i : i + len(right)] += left[i] * right
    return conv_result


def plot_soundwave(ax, indx_ax: int, sample_data: np.array, sample_rate: int):
    """Plot the wave of the sound.

    Args:
        ax (list): The place to plot.
        indx_ax (int): The number of ax.
        sample_data (np.array): The sound data which plot.
        sample_rate (int): Sample rate of the sample_data.
    """
    # 音源波形を表示
    N = len(sample_data)  # サンプル数を出しておく
    time = np.arange(0, N) / sample_rate  # サンプルレートで割って時間に変換
    ax[indx_ax].plot(time, sample_data)
    ax[indx_ax].set_xlabel("Time")
    ax[indx_ax].set_ylabel("Magnitude")


def spectrogram(ax, fig, indx_ax: int, data: np.ndarray, sample_rate: int, F_size=1024):
    """Calculate STFT and show the spectrogram.

    Args:
        ax (list): The place to plot.
        fig: The figure of plt.
        indx_ax (int): The number of ax.
        data (np.ndarray): The data of the sound.
        sample_rate (int): The sample rate of sample_data.
        F_size (int, optional): The size of frame. Defaults to 1024.
    """
    f, t, spec = scipy.signal.spectrogram(
        data, sample_rate, nperseg=F_size
    )  # 短時間フーリエ変換して振幅成分を取り出す
    ax_spec = ax[indx_ax].pcolormesh(
        t, f, 20 * np.log10(spec), vmax=1e-6, cmap="CMRmap"
    )  # dbに変換してスペクトログラムを表示
    ax[indx_ax].set_xlabel("Time [sec]")
    ax[indx_ax].set_ylabel("Fequency [Hz]")
    fig.colorbar(ax_spec, ax=ax[indx_ax], aspect=5, location="right")


def highpass_filer(
    div_num: int, cut_off_frequency: int, sample_rate: int, window=np.hamming
):
    """Make high-pass filter.

    Args:
        div_num (int): The number of the division.
        cut_off_frequency (int): The cut-off frequency.
        sample_rate (int): The sample rate of the sound.
        window (numpy, optional): The setting of the window. Defaults to np.hamming.

    Returns:
        ndarray: The high-pass filter.
    """
    cut_off_fq_to_time = 2 * cut_off_frequency / (sample_rate)  # カットオフ周波数をsinc関数軸に変換
    x = np.array([t for t in range(-div_num // 2, div_num // 2)])
    hpf = (np.sinc(x) - np.sinc(cut_off_fq_to_time * x) * cut_off_fq_to_time) * window(
        div_num
    )
    return np.array(hpf)


def main():
    """Do main action."""
    # 初期値の設定
    fig, ax = plt.subplots(4, 1, layout="constrained", sharex=True)
    fig2, ax2 = plt.subplots(3, 1, layout="constrained")
    div_num = 201  # フィルターの分割数
    cut_off_frequency = 5000  # 遮断周波数

    # 音源の読み込み、出力
    sample_path = "sample.wav"  # 音源へのパスを指定
    sample_data, sample_rate = soundfile.read(sample_path)  # 音源データを取得(numpy型)
    ax[0].set_title("Original wave")
    plot_soundwave(ax, 0, sample_data, sample_rate)  # フィルタリング前の波形を出力
    ax[2].set_title("Original spectrogram")
    spectrogram(ax, fig, 2, sample_data, sample_rate)  # フィルタリング前のスペクトログラムを出力

    # フィルタリング処理
    hpf = highpass_filer(div_num, cut_off_frequency, sample_rate)  # ハイパスフィルタの作成
    hpf_fft = np.fft.fft(hpf)  # ハイパスフィルタを周波数変換
    freq = np.fft.fftfreq(div_num, d=1.0 / sample_rate)

    # ハイパスフィルタの情報を出力
    ax2[0].set_title("High-pass filter (Magnitude)")
    ax2[0].set_xlabel("Frequency[Hz]")
    ax2[0].set_ylabel("Amplitude[db]")
    ax2[0].plot(
        freq[: div_num // 2], np.abs(hpf_fft[: div_num // 2])
    )  # ハイパスフィルタの振幅情報を出力
    ax2[1].set_title("High-pass filter (Angle)")
    ax2[1].set_xlabel("Frequency[Hz]")
    ax2[1].set_ylabel("Angle[rad]")
    ax2[1].plot(
        freq[: div_num // 2], np.unwrap(np.angle(hpf_fft[: div_num // 2]))
    )  # ハイパスフィルタの位相情報を出力
    ax2[2].set_title("Filter preview")
    ax2[2].set_xlabel("The number of the sample")
    ax2[2].set_ylabel("Amplitude[db]")
    ax2[2].plot(np.arange(div_num), hpf)  # ハイパスフィルタの時間状態を出力

    # ハイパスフィルタを適用、結果を出力
    conv_result = conv(sample_data, hpf)  # 音源とフィルタを畳み込み
    ax[1].set_title("Filtered wave")
    plot_soundwave(ax, 1, conv_result, sample_rate)  # フィルタリング後の波形を出力
    ax[3].set_title("Filtered spectrogram")
    spectrogram(ax, fig, 3, conv_result, sample_rate)  # フィルタリング後のスペクトログラムを出力

    plt.show()
    fig.savefig("ex2_sample.png")
    fig2.savefig("ex2_highpass_filter.png")
    plt.clf()
    plt.close()
    soundfile.write(
        file="highpass_sample.wav", data=conv_result, samplerate=sample_rate
    )


if "__main__" == __name__:
    main()
