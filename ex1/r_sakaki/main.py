import matplotlib.pyplot as plt
import matplotlib as mpl
import librosa
import numpy as np



def load_file(filename):
    y, sr = librosa.load(filename, sr=None)
    totaltime = len(y)/sr
    time_array = np.arange(0, totaltime, 1/sr)
    return sr, y, time_array


def stft(data, win_size, overlap):
    spec = []
    win_func = np.hamming(win_size)
    step = win_size-overlap

    for i in range(int((data.shape[0]-overlap)/step)):
      tmp = data[i*step: i*step+win_size] * win_func  # 窓関数をつける
      tmp_fft = np.fft.rfft(tmp)    # フーリエ変換
      spec.append(tmp_fft)
    return np.transpose(spec)


def istft(spec, win_size, overlap):
    win_func = np.hamming(win_size)
    step = win_size-overlap
    spec = np.transpose(spec)

    tmp_istft = np.fft.irfft(spec)    # 逆変換
    tmp_istft = tmp_istft / win_func  # 窓関数を外す
    tmp_istft = tmp_istft[:, :step]   # オーバーラップを外す
    amp = tmp_istft.reshape(-1)
    return amp
    

# 音声の読込
samplerate, data, time_array = load_file("sample.mp3")

# 窓関数とオーバーラップの設定
win_size=1024
overlap=512

# 短時間フーリエ変換
spectrogram = stft(data, win_size, overlap)
spectrogram_db = 20*np.log(np.abs(spectrogram))   # db変換

# フーリエ逆変換
new_data = istft(spectrogram, win_size, overlap)


# グラフの表示の準備
fig, ax = plt.subplots(3, 1, figsize=(10,10))
fig.subplots_adjust(hspace=0.5)

# 元の音声の波形を表示
librosa.display.waveshow(data, sr=samplerate, ax=ax[0])
ax[0].set(title="Original Signal", ylabel="Amplitude", xlim=[0, time_array[-1]])

# スペクトログラムの表示
img = librosa.display.specshow(
    spectrogram_db,
    sr=samplerate,
    hop_length=win_size-overlap,
    ax=ax[1],
    x_axis="time",
    y_axis="log",
    cmap="jet",
)
ax[1].set(title="Spectrogram", xlabel="Time [sec]", ylabel="Freq [Hz]", xlim=[0, time_array[-1]])
fig.colorbar(img, ax=ax[1], format="%+2.f dB")

# 逆変換で得た波形の表示
librosa.display.waveshow(new_data, sr=samplerate, ax=ax[2])
ax[2].set(title="Signal", ylabel="Amplitude", xlim=[0, time_array[-1]])


plt.show()
plt.close()
