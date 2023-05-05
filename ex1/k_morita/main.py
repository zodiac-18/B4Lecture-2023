"""Main."""
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import spectrogram as sp

if __name__ == "__main__":
    """main"""
    soundfile = sys.argv[1]
    samplerate = 16000
    window_length = 512
    overlap_rate = 0.5

    # load file
    y, samplerate = librosa.load(soundfile, sr=samplerate)
    nyquist_freq = samplerate // 2

    # stft
    fft_array, total_time, total_frame = sp.stft(
        y, sr=samplerate, win_len=window_length, ol=overlap_rate
    )
    mag, _ = sp.magphase(fft_array)
    mag_db = sp.mag_to_db(mag)

    # istft
    y_inv, _, _ = sp.istft(
        fft_array, sr=samplerate, win_len=window_length, ol=overlap_rate
    )

    # draw graph
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    x = np.linspace(0, total_time, total_frame)

    # draw input wave
    axes[0].plot(x, y[:total_frame])
    axes[0].set_title("input wave")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].set_xlim(0, total_time)

    # draw spectrogram
    im = axes[1].imshow(
        mag_db,
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

    # draw restoration wave
    axes[2].plot(x, y_inv[:total_frame])
    axes[2].set_title("restoration wave")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Magnitude [dB]")
    axes[2].set_xlim(0, total_time)

    plt.tight_layout()
    plt.savefig("result.png")

    # save restoration wave as wav
    sf.write("audio_inv.wav", data=y_inv, samplerate=samplerate)
