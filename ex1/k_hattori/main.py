import numpy as np
import matplotlib.pyplot as plt
import wave


def main():
    # wavファイルの読み込み
    wavf = 'ONSEI.wav'
    wr = wave.open(wavf, 'r')
    ch = wr.getnchannels()  # チャンネル数
    samplesize = wr.getsampwidth()   # サンプルサイズ
    fr = wr.getframerate()  # サンプリングレート
    fn = 436000    # データ整形のために指定
    data = wr.readframes(fn)    # fn個のフレームを読んでbytesオブジェクトで返す
    wr.close()
    # 数値に変換
    if samplesize == 2:
        num_data = np.frombuffer(data, dtype = 'int16')
    elif samplesize == 4:
        num_data = np.frombuffer(data, dtype = 'int32')
    # データを扱いやすく整形
    num_data = num_data[:fn]/float((2^15))
    TOTAL_TIME = 1.0 * fn / fr

    time = np.linspace(0, TOTAL_TIME, num_data.shape[0])
    sampling_rate = fr

    print('チャンネル', ch)
    print('総フレーム数', fn)
    print('再生時間', TOTAL_TIME)
    print(num_data)

    plt.plot(time, num_data)
    plt.xlabel("time [s]")
    plt.ylabel("Amplitude")
    plt.show()

    WIDTH = 1000    # 分割フレームの大きさ
    OVERLAP = int(WIDTH / 2) # フレームのシフト幅

    wave_length = num_data.shape[0]    # 音声の全フレーム数
    split_number = len(np.arange((WIDTH/2), wave_length, (WIDTH - OVERLAP))) # 音声の分割数
    unit_time = TOTAL_TIME * WIDTH / fn # フレーム時間分解能
    unit_freq = 1 / unit_time   # フレーム周波数分解能

    print(unit_time, unit_freq)

    window = np.hamming(WIDTH)
    pos = 0

    # 分割でフーリエ変換したデータのサイズを求める
    fframe_size = np.fft.fft(num_data[pos:pos+WIDTH])
    fframe_size = fframe_size.shape[0]
    # スペクトログラムの配列
    spec = np.zeros([split_number, fframe_size], dtype=complex)
    im_result = spec    # 虚部の確保をする配列

    # STFT
    for i in range(split_number):
        frame = num_data[pos:pos+WIDTH]
        windowed = window * frame   # 窓関数をかける
        # 短時間に分けたものをフーリエ変換
        fft_result = np.fft.fft(windowed)

        spec[i] = fft_result
        pos += OVERLAP
    
    # グラフにプロットするために実数の対数をとる
    amp = np.real(spec[:,int(spec.shape[1]/2)::-1])
    amp = np.log(amp** 2)

    # スペクトログラムの表示
    plt.rcParams['image.cmap'] = 'jet'
    plt.imshow(amp.T, extent=[0, TOTAL_TIME, 0, sampling_rate/2000], aspect="auto")
    plt.xlabel("time [s]")
    plt.ylabel("frequency [kHz]")
    plt.colorbar()
    plt.show()

    ifft_wave = np.zeros(num_data.shape)
    pos = 0
    # iSTFT
    for i in range(split_number):
        ifft_result = np.fft.ifft(spec[i])
        windowed = np.real(ifft_result)
        ifft_wave[pos:pos+WIDTH] = windowed / window
        
        pos += OVERLAP

    plt.plot(time, ifft_wave)
    plt.show()




if __name__ == '__main__':
    main()
    exit(1)

