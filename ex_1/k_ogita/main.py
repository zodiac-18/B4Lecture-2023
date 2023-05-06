#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def stft(data, framesize, overlap):
    """_summary_

    Args:
        data (_type_): _description_
        time (_type_): _description_
        framesize (_type_): _description_
        samplerate (_type_): _description_
        overlap (_type_): _description_

    Returns:
        ndarray: 
    """
    window = np.hamming(framesize)
    split_time = int(data.shape[0]//(framesize*(1-overlap)))-1
    stft_result = []
    pos = 0
    for _ in range(split_time):
        frame = np.fft.fft(data[int(pos):int(pos+framesize)]*window)
        stft_result.append(frame)
        pos += framesize * (1 - overlap)
    return np.array(stft_result)

def istft(spec, framesize, overlap):
    window = np.hamming(framesize)
    # calculate the number of wave samples
    num_istft = spec.shape[0] * framesize*(1-overlap) + framesize
    istft_result = np.zeros(int(num_istft))
    pos = 0
    for i in range(spec.shape[0]):
        frame = np.fft.ifft(spec[i,:])
        frame = np.real(frame) * window
        istft_result[int(pos):int(pos+framesize)] += frame
        pos += framesize * (1 - overlap)
    print(istft_result)
    return istft_result

def main():
    sound_file = 'miku.wav'
    framesize = 1024
    overlap = 0.5
    data, samplerate = sf.read(sound_file)
    time = len(data)/samplerate
    spectrogram = stft(data, framesize, overlap)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    # 横軸
    t_1 = np.arange(0,len(data)) / samplerate
    ax1.plot(t_1, data)
    plt.title("Original signal")
    plt.xlabel("Time[s]")
    plt.ylabel("Magnitude")
    # plot spectrogram
    spectrogram_amp = np.log(np.abs(spectrogram[:,:int(framesize*(1-overlap))]))
    ax2 = fig.add_subplot(3,1,2)
    ax2.set_title("Spectrogram")
    # スペクトログラムで縦軸周波数、横軸時間にするためにデータを転置
    im = ax2.imshow(spectrogram_amp.T, extent=[0, time, 0, samplerate/2000]
                    , aspect='auto')
    ax2.set_ylabel("Frequency [kHz]")
    ax2.set_xlim(0, time)
    ax2.set_ylim(0, samplerate/2000)
    fig.colorbar(im, ax=ax2, format="%+2.f dB")
    
    istft_wave = istft(spectrogram, framesize, overlap)
    ax3 = fig.add_subplot(3,1,3)
    # 横軸
    t_3 = np.arange(0,len(istft_wave)) / samplerate
    ax3.plot(t_3, istft_wave)
    plt.title("Re-Synhesized signal")
    plt.xlabel("Time[s]")
    plt.ylabel("Magnitude")
    plt.savefig("wave.png")
    plt.show()
    plt.close() 

    
if "__main__" == __name__:
    main()