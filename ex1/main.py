import argparse
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.fftpack as fftpack
import scipy.signal as signal
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Generate spectrogram and inverse transform')
    parser.add_argument('--input-file', type=str, required=True, help='input wav file')
    parser.add_argument('--nfft', type=int, default=1024, help='number of FFT points')
    parser.add_argument('--hop-length', type=int, default=512, help='number of samples between successive STFT columns')
    parser.add_argument('--window', type=str, default='hann', help='window function type')
    return parser.parse_args()

def main():
    args = parse_args()

    # Read input wav file
    rate, data = wavfile.read(args.input_file)
    data = np.array(data, dtype=float)

    """波形をプロットする"""
    time = np.arange(0, len(data)) / rate
    plt.plot(time, data)
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    #plt.show()
    
    # STFT parameters
    nfft = args.nfft
    hop_length = args.hop_length
    window = args.window
    window_func = signal.get_window(window, nfft)
    print(f'window_func:{window_func}')

    # Compute spectrogram
    spectrogram = np.zeros((1 + nfft // 2, (len(data) - nfft) // hop_length + 1), dtype=np.complex128)
    print(f'spectrogram:{spectrogram}')
    for i in range(spectrogram.shape[1]):
        segment = data[i * hop_length:i * hop_length + nfft] * window_func
        spectrum = fftpack.fft(segment, n=nfft, axis=0)[:1 + nfft // 2]
        spectrogram[:, i] = spectrum
    print(f'data:{data}')
    print(f'segment:{segment}')
    print(f'spectrum:{spectrum}')
    print(f'spectrogram:{spectrogram}')
    # Plot spectrogram
    plt.figure()
    plt.imshow(20 * np.log10(np.abs(spectrogram)), origin='lower', aspect='auto', cmap='jet')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar()
    plt.title('Spectrogram')
    #plt.show()

    import pdb; pdb.set_trace()

    # Compute inverse STFT
    data_inv = np.zeros_like(data)
    window_sum = np.zeros_like(data_inv)
    for i in range(spectrogram.shape[1]):
        spectrum = spectrogram[:, i]
        segment = fftpack.ifft(np.concatenate((spectrum, np.conj(spectrum[-2:0:-1]))), axis=0).real
        segment = segment[:hop_length] if i == 0 else segment[hop_length:-hop_length]
        data_inv[i * hop_length:i * hop_length + nfft] += segment
        window_sum[i * hop_length:i * hop_length + nfft] += window_func ** 2
    
    data_inv /= window_sum
    data_inv = np.int16(data_inv)


if __name__ == '__main__':
    main()
