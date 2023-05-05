#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def stft(data, framesize, overlap, window):
    frame_length = data[0]
    
    
    
def main():
    sound_file = 'ex1.wav'
    framesize = 2048
    overlap = 0.8
    window = np.hamming(framesize)
    data, samplerate = sf.read(sound_file)
    time = len(data)/samplerate
    
    stft_wave = stft(data, framesize, overlap, window)
    