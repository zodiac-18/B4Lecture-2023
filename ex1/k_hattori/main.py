import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from mpl_toolkits.axes_grid1 import make_axes_locatable


def wavload(path):
    """"引数のpathから音声を読み込み，データとサンプリングレートを返す"""
    data, samplerate = sf.read(path)
    return data, samplerate


def main():
    # wavファイルの読み込み
    path = 'ONSEI.wav'
    data, samplerate = wavload(path)
    fn = 436000    # データ整形のために指定

    # データを扱いやすく整形
    data = data[:fn]
    time = np.arange(0, len(data))/samplerate
    TOTAL_TIME = len(data)/samplerate
    WIDTH = 1000    # 分割フレームの大きさ
    OVERLAP = int(WIDTH / 2) # フレームのシフト幅

    wave_length = data.shape[0]    # 音声の全フレーム数
    split_number = len(np.arange((WIDTH/2), wave_length, (WIDTH - OVERLAP))) 
    # 音声の分割数

    # 分割でフーリエ変換したデータのサイズを求める
    fframe_size = np.fft.fft(data[:WIDTH])
    fframe_size = fframe_size.shape[0]
    # STFTの格納配列
    spec = np.zeros([split_number, fframe_size], dtype=complex)

    window = np.hamming(WIDTH) # ハミング窓
    pos = 0 # 窓を掛ける位置

    # STFT
    for i in range(split_number):
        frame = data[pos:pos + WIDTH]
        windowed = window * frame   # 窓関数をかける
        # 短時間に分けたものをフーリエ変換
        fft_result = np.fft.fft(windowed)

        spec[i] = fft_result
        pos += OVERLAP
    
    # グラフにプロットするために実数の対数をとる
    amp = np.real(spec[:,int(spec.shape[1]/2)::-1])
    amp = np.log(amp** 2)

    ifft_wave = np.zeros(data.shape)
    pos = 0
    # iSTFT
    for i in range(split_number):
        ifft_result = np.fft.ifft(spec[i])
        windowed = np.real(ifft_result)
        ifft_wave[pos:pos+WIDTH] = windowed / window
        
        pos += OVERLAP

    
    # 時間信号，スペクトログラム，逆変換後をグラフにプロット
    fig = plt.figure(figsize=(6,6))
    plt.rcParams["image.cmap"] = "jet"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    ax1 = fig.add_subplot(311)
    ax1.set_title("Original signal")
    ax1.plot(time, data)
    ax1.set_ylabel("Magnitude")
    ax1.set_ylim(-1,1)
    ax1.set_xlim(0,TOTAL_TIME)

    ax2 = fig.add_subplot(312)
    ax2.set_title("Spectrogram")
    im = ax2.imshow(amp.T, extent=[0, TOTAL_TIME, 0, samplerate/2000]
                    , aspect='auto')
    divider1 = make_axes_locatable(ax2)
    cax1 = divider1.append_axes("right", size="2%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax1)
    ax2.set_ylabel("Frequency [kHz]")
    ax2.set_xlim(0, TOTAL_TIME)
    ax2.set_ylim(0, samplerate/2000)

    ax3 = fig.add_subplot(313)
    ax3.set_title("Re-synthesized signal")
    ax3.plot(time, ifft_wave)
    ax3.set_ylabel("Magnitude")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylim(-1,1)
    ax3.set_xlim(0,TOTAL_TIME)

    fig.tight_layout()

    # グラフの横幅を揃える
    fig.canvas.draw()
    axpos1 = ax1.get_position() # 一段目の図の描画領域
    axpos2 = ax2.get_position() # 二段目の図の描画領域
    axpos3 = ax3.get_position() # 三段目の図の描画領域
    #ax1とax3の幅をax2と同じにする
    ax1.set_position([axpos1.x0, axpos1.y0, axpos2.width, axpos1.height])
    ax3.set_position([axpos3.x0, axpos3.y0, axpos2.width, axpos3.height])

    plt.show()


if __name__ == '__main__':
    main()
    exit(1)

