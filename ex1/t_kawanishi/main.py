import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

SOUND_PATH = "voice.wav"                     # path to the sound
Fs = 512                                    # frame size
overlap_r = 0.9                              # overlap rate between 0 to 1


# load sound file
def load_sound(sound_path):
    data, samplerate = sf.read(file=sound_path)
    return data, samplerate

# short-time fourier transform
def stft (data, overlap, Fs, samplerate):

    frame_dist = int(Fs*(1-overlap))                       # distance between segment and segment
    frame_group = []                                       # list to collect frame
    win = np.hamming(Fs)                                   # create windows

    # loop to cut data into segment and adapt hamming window & fft
    frame_s = 0
    while True:
        if frame_s+Fs > data.shape[0]:
            break   
        frame = np.fft.fft(data[frame_s:frame_s+Fs]*win)
        frame_group.append(frame)
        frame_s += frame_dist
    freq = np.fft.fftfreq(Fs, d=1/samplerate)
    freq = freq/1000
    frame_l = frame_s                                       # last frame's location
    return frame_group, freq, frame_l

# inverse stft
def istft(data, overlap, length):

    
    # create a list to generate sound data
    origin_sound = np.zeros(length)
    seg_s = 0
    Fs = len(data[0])
    # using ifft to inverse segment
    for i in range(len(data)):
        origin_sound[seg_s:seg_s+Fs] +=np.real(np.fft.ifft(data[i]))
        seg_s += int(Fs*(1-overlap))
    
    return origin_sound
    


data, samplerate = load_sound(SOUND_PATH)         

frame_group2, freq, frame_l = stft(data, overlap_r, Fs, samplerate)
origin_sound = istft(frame_group2, overlap_r, data.shape[0])

# compute time
x_t = np.arange(0,len(data))/samplerate
spec_t = np.arange(start=0, stop=frame_l, step=int(Fs*(1-overlap_r)))/samplerate

frame_positive = np.delete(frame_group2, slice(int(Fs/2)-1,Fs),1)                                           #cut negative data
frame_positive = frame_positive.T

# create graph's group
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# create subplot domain
base_sound = axs[0]
sound_spec = axs[1]
repro_sound = axs[2]
fig.subplots_adjust(hspace=0.5)

#set x limit
base_sound.set_xlim(0,len(data)/samplerate)
repro_sound.set_xlim(0,len(data)/samplerate)

# plot data
base_sound.plot(x_t, data)
cax = fig.add_axes([0.92, 0.395, 0.02, 0.2]) # x, y, width, height
spec_d = sound_spec.pcolormesh(spec_t, freq[1:int(Fs/2)], 10*np.log(np.abs(frame_positive)), cmap='plasma', shading='nearest')
color_b = fig.colorbar(spec_d, cax=cax)
color_b.set_label('Amplitude[dB]',labelpad=-0.1)
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


