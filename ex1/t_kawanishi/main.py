"""To generate spectrogram."""
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='This is a program to generate sound\'s spectrogram')
parser.add_argument('arg1',help='the path of sound')
parser.add_argument('--arg2',help='frame size recommend between 128 to 2048 ', default= 512)
parser.add_argument('--arg3',help='overlap rate between 0 to 1', default=0.5)
parser.add_argument('--arg4',help='limit of spectrogram\'s frequency', default=25000)

def load_sound(sound_path):
    """Load sound file."""
    data, samplerate = sf.read(file=sound_path)
    return data, samplerate


def stft(data, overlap, Fs, samplerate):
    """Short-time fourier transform."""
    frame_dist = int(Fs * (1 - overlap))                   # distance between segment and segment
    frame_group = []                                       # list to collect frame
    win = np.hamming(Fs)                                   # create windows

    # loop to cut data into segment and adapt hamming window & fft
    frame_s = 0
    while True:
        if frame_s + Fs > data.shape[0]:
            break
        frame = np.fft.fft(data[frame_s:frame_s + Fs] * win)
        frame_group.append(frame)
        frame_s += frame_dist
    freq = np.fft.fftfreq(Fs, d=1 / samplerate)
    return frame_group, freq, frame_s


def istft(data, overlap, length):
    """Inverse stft."""
    # create a list to generate sound data
    origin_sound = np.zeros(length)
    seg_s = 0
    Fs = len(data[0])
    # using ifft to inverse segment
    for i in range(len(data)):
        origin_sound[seg_s:seg_s + Fs] += np.real(np.fft.ifft(data[i]))
        seg_s += int(Fs * (1 - overlap))
    return origin_sound


if __name__ == '__main__':

    args = parser.parse_args()
    Fs = args.arg2
    overlap_r = args.arg3
    spec_ylim = args.arg4
    data, samplerate = load_sound(args.arg1)  # arg1 is sound path
    frame_group2, freq, frame_l = stft(data, overlap_r, Fs, samplerate)
    origin_sound = istft(frame_group2, overlap_r, data.shape[0])

    # compute time
    x_t = np.arange(0, len(data)) / samplerate
    spec_t = np.arange(start=0, stop=frame_l, step=int(Fs * (1 - overlap_r))) / samplerate

    frame_positive = np.delete(frame_group2, slice(int(Fs / 2) - 1, Fs), 1)         # cut negative data
    frame_positive = frame_positive.T

    # create graph's group
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # create subplot domain
    base_sound = axs[0]
    sound_spec = axs[1]
    repro_sound = axs[2]
    fig.subplots_adjust(hspace=0.5)

    # set x limit
    base_sound.set_xlim(0, len(data) / samplerate)
    repro_sound.set_xlim(0, len(data) / samplerate)

    # set y limit
    if spec_ylim <= max(freq):      # if maximum freq bigger than limit then set y-axis limit
        sound_spec.set_ylim(0,spec_ylim)

    # plot data
    base_sound.plot(x_t, data)
    cax = fig.add_axes([0.92, 0.395, 0.02, 0.2])    # x, y, width, height
    spec_d = sound_spec.pcolormesh(spec_t, freq[1:int(Fs / 2)],
                                10 * np.log(np.abs(frame_positive)),
                                cmap='plasma', shading='nearest')
    color_b = fig.colorbar(spec_d, cax=cax)
    color_b.set_label('Amplitude[dB]', labelpad=-0.1)
    repro_sound.plot(x_t, origin_sound)

    # set x label
    base_sound.set_xlabel('Time[s]')
    sound_spec.set_xlabel('Time[s]')
    repro_sound.set_xlabel('Time[s]')

    # set y label
    base_sound.set_ylabel('Sound pressure[Pa]')
    sound_spec.set_ylabel('Frequency[kHz]')
    repro_sound.set_ylabel('Sound pressure[Pa]')

    # set title
    base_sound.set_title('Original signal')
    sound_spec.set_title('Spectrogram')
    repro_sound.set_title('Re-synthesized signal')

    plt.show()
    fig.savefig('comparision_graph_ex1.png')
