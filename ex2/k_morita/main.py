"""Main."""
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import filter as f
import spectrogram as sp

if __name__ == "__main__":
    """main"""
    soundfile = sys.argv[1]
    window_length = 512
    overlap_rate = 0.5

    # load file
    y, samplerate = librosa.load(soundfile, sr=None)
    nyquist_freq = samplerate // 2

    filter = f.create_lpf(2000, samplerate, 101)
    filtered_y = f.conv(y, filter)

    # stft
    fft_array, total_time, total_frame = sp.stft(
        filtered_y, sr=samplerate, win_len=window_length, ol=overlap_rate
    )
    mag, _ = sp.magphase(fft_array)
    mag_db = sp.mag_to_db(mag)

    # draw graph
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    x = np.linspace(0, total_time, total_frame)

    # draw input wave
    axes[0].plot(x, y[:total_frame])
    axes[0].set_title("input wave")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].set_xlim(0, total_time)

    # draw input spectrogram
    ifft, _, _ = sp.stft(y, samplerate, win_len=window_length, ol=overlap_rate)
    imag, _ = sp.magphase(ifft)
    im = axes[1].imshow(
        sp.mag_to_db(imag),
        cmap="jet",
        aspect="auto",
        vmin=-60,
        vmax=30,
        extent=[0, total_time, 0, nyquist_freq],
    )
    axes[1].set_title("spectrogram")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_xlim(0, total_time)
    axes[1].set_ylim(0, nyquist_freq)
    plt.colorbar(mappable=im, ax=axes[1], orientation="horizontal")

    # draw filtered wave
    axes[2].plot(x, filtered_y[:total_frame])
    axes[2].set_title("filtered wave")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Magnitude [dB]")
    axes[2].set_xlim(0, total_time)

    # draw spectrogram
    im = axes[3].imshow(
        mag_db,
        cmap="jet",
        aspect="auto",
        vmin=-60,
        vmax=30,
        extent=[0, total_time, 0, nyquist_freq],
    )
    axes[3].set_title("spectrogram")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel("Frequency [Hz]")
    axes[3].set_xlim(0, total_time)
    axes[3].set_ylim(0, nyquist_freq)
    plt.colorbar(mappable=im, ax=axes[3], orientation="horizontal")

    plt.tight_layout()
    plt.savefig("result.png")

    # save restoration wave as wav
    sf.write("audio_filtered.wav", data=filtered_y, samplerate=samplerate)
