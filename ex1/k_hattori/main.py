import numpy as np
import matplotlib.pyplot as plt
import wave


def main():
    # wavファイルの読み込み
    wavf = 'ONSEI.wav'
    wr = wave.open(wavf, 'r')
    ch = wr.getnchannels()  # チャンネル数
    width = wr.getsampwidth()   # サンプルサイズ
    fr = wr.getframerate()  # サンプリングレート
    fn = wr.getnframes()    # フレーム数
    data = wr.readframes(fn)    # fn個のフレームを読んでbytesオブジェクトで返す
    TOTAL_TIME = 1.0 * fn / fr
    wr.close()
    # 数値に変換
    num_data = np.frombuffer(data, dtype = 'int16')
    time = np.linspace(0, TOTAL_TIME, num_data.shape[0])
    sampling_rate = fr

    print('チャンネル', ch)
    print('総フレーム数', fn)
    print('再生時間', TOTAL_TIME)
    print(num_data)

    plt.plot(time, num_data)
    plt.show()

    # データを扱いやすく整形
    num_data = num_data[:436000]/float((2^15))

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
    fframe_size = fframe_size[:int(len(fframe_size)/2)]
    fframe_size = fframe_size.shape[0]
    # スペクトログラムの配列
    spec = np.zeros([split_number, fframe_size])
    im_result = spec    # 虚部の確保をする配列

    # STFT
    for i in range(split_number):
        frame = num_data[pos:pos+WIDTH]
        windowed = window * frame   # 窓関数をかける
        # フーリエ変換．折り返しをカットして対数をとる
        fft_result = np.fft.fft(windowed)
        # plt.plot(windowed)
        # plt.show()
        # plt.plot(fft_result)
        # plt.show()
        f_frame = np.real(fft_result[:int(len(fft_result)/2)])
        f_frame = np.log(f_frame ** 2)

        spec[i] = f_frame[::-1]

        pos += OVERLAP
    



    # print(spec)
    plt.rcParams['image.cmap'] = 'jet'
    plt.imshow(spec.T, extent=[0, TOTAL_TIME, 0, sampling_rate/2000], aspect="auto")
    plt.xlabel("time [s]")
    plt.ylabel("frequency [kHz]")
    plt.colorbar()
    plt.show()









if __name__ == '__main__':
    main()
    exit(1)

