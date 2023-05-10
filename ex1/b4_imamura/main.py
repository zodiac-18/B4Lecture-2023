"""音声の短時間フーリエ変換・逆変換を実装."""

import copy

import matplotlib.pylab as plt
import numpy as np
import scipy.signal as sp
import soundfile


def main():
    """Do main action."""
    # 音源の読み込み
    sample_path = "sample.wav"  # 音源へのパスを指定
    sample_data, sample_rate = soundfile.read(sample_path)  # 音源データを取得(numpy型)
    N = len(sample_data)  # サンプル数を出しておく
    # サンプルレートは44100Hz

    # 音源波形を表示
    time = np.arange(0, N) / sample_rate  # サンプルレートで割って時間に変換
    fig, ax = plt.subplots(3, 1, layout="constrained", sharex=True)
    ax[0].set_title("Original Wave")
    ax[0].plot(time, sample_data)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Magnitude")

    # 短時間フーリエ変換
    fft = []  # fftの結果を格納する
    F_size = 1024  # 切り出すフレーム幅
    OverRap = 512  # オーバーラップ率50%
    dist_of_Frame = F_size - OverRap  # フレーム間の距離
    win = sp.hann(F_size)  # 窓関数としてハニング窓を使用(他にハミング窓、ブラックマン窓などが存在)
    fft_start = 0  # フレームの開始位置
    while fft_start + F_size <= N:  # 次のフレーム幅の末端がサンプル数を超えるまでfft
        Frame = sample_data[fft_start : fft_start + F_size]  # 音源データからフレームを切り出す
        fft_per = np.fft.fft(win * Frame)  # fftを実行
        fft.append(fft_per)  # 結果を格納
        fft_start += dist_of_Frame
    fft_for_irfft = copy.deepcopy(fft)  # 逆変換用にコピー
    fft = np.array(fft)
    fft_abs = np.abs(fft[:, :dist_of_Frame]).T  # オーバーラップ部分を除きfftの結果全ての振幅を出す

    # fftの結果のスペクトログラムを表示
    fft_spec = 20 * (np.log10(fft_abs))  # スペクトログラムを見やすくするための処理
    ax1 = ax[1].imshow(
        fft_spec,
        origin="lower",
        aspect="auto",
        extent=(0, N / sample_rate, 0, sample_rate // 2),
        cmap="plasma",
    )
    ax[1].set_title("Spectrogram")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Frequency")
    fig.colorbar(ax1, ax=ax[1], aspect=5, location="right")
    # extentでxは0 ~ サンプル数/サンプルレート(秒数)、yは0 ~ サンプルレート/2(Hz)までを表示

    # フーリエ逆変換
    ifft = np.zeros(N)  # 逆変換の結果格納
    start_ifft = 0  # 逆変換した波形を格納するスタート位置
    for fft_per in fft_for_irfft:
        ifft_per = np.fft.ifft(fft_per) / win  # 逆変換
        ifft[start_ifft : start_ifft + dist_of_Frame] += np.real(
            ifft_per[:dist_of_Frame]
        )  # 逆変換結果を格納
        start_ifft += dist_of_Frame

    # 逆変換の結果の波形を表示
    plt.title("Re-synthesized Wave")
    ax[2].plot(time, ifft)
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Magnitude")

    plt.show()
    fig.savefig("ex1.png")
    plt.clf()
    plt.close()
    soundfile.write(file="Ifft_voice.wav", data=ifft, samplerate=sample_rate)


if "__main__" == __name__:
    main()
