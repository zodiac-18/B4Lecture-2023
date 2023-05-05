#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def stft(data, time, framesize, samplerate, overlap):
    frame_cycle = framesize/samplerate
    window = np.hamming(framesize)
    split_time = int((time-frame_cycle*overlap)/frame_cycle*(1-overlap))
    
    pos = 0
    for _ in range(split_time):
        frame = np.fft.fft(data[pos:pos+framesize]*window)
        stft_result.append(frame)
        pos += framesize * (1 - overlap)
    return stft_result

    
def main():
    sound_file = 'ex1.wav'
    framesize = 2048
    overlap = 0.8
    data, samplerate = sf.read(sound_file)
    time = len(data)/samplerate
    
    stft_wave = stft(data, time, framesize, samplerate, overlap)
    
    
if "__main__" == __name__:
    main()