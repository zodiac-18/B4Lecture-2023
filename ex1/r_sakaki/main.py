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
      tmp = data[i*step: i*step+win_size] * win_func
      tmp_fft = np.fft.rfft(tmp)
      spec.append(tmp_fft)
    return np.transpose(spec)


def istft(spec, win_size, overlap):
    win_func = np.hamming(win_size)
    step = win_size-overlap
    spec = np.transpose(spec)

    tmp_istft = np.fft.irfft(spec)
    tmp_istft = tmp_istft / win_func
    tmp_istft = tmp_istft[:, :step]
    amp = tmp_istft.reshape(-1)
    return amp
    


# 音声の読込
samplerate, data, time_array = load_file("taiko.mp3")

# 短時間フーリエ変換
### 結果の画像を見るとすぐわかるのですが、マイナスの値が入ってしまっています。
### どこかが違うと思うのですが、それがわからないです。
### よろしくお願いします
win_size=256
overlap=128
spectrogram = stft(data, win_size, overlap)
spectrogram_db = 20*np.log(np.abs(spectrogram))
# spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

# フーリエ逆変換
new_data = istft(spectrogram, win_size, overlap)

# 各グラフの表示
fig, ax = plt.subplots(3, 1, figsize=(10,10))
fig.subplots_adjust(hspace=0.5)

# 元の音声を表示
librosa.display.waveshow(data, sr=samplerate, ax=ax[0])
ax[0].set(title="Original Signal", ylabel="Amplitude", xlim=[0, time_array[-1]])

# スペクトログラムの表示
img = librosa.display.specshow(
    spectrogram_db,
    sr=samplerate,
    hop_length=128,
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
