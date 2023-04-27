import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


def stft(waveform, frame_size=1024, frame_shift=256):
    """
    Compute the Short-Time Fourier Transform (STFT) of a waveform.

    Args:
        waveform (ndarray): Input waveform.
        frame_size (int): Size of each frame. Defaults to 1024.
        frame_shift (int): Number of samples to shift between frames. Defaults to 256.

    Returns:
        ndarray: Spectrogram of the waveform.
    """
    # Get the number of samples of the waveform
    num_samples = waveform.shape[0]

    # Calculate the number of frames and FFT size
    num_frames = (num_samples - frame_size) // frame_shift + 1
    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    # Make an empty array for the spectrogram
    spectrogram = np.zeros((num_frames, frame_size))
    for frame_idx in range(num_frames):
        start_idx = frame_idx * frame_shift
        end_idx = start_idx + frame_size

        # Extract the current frame from the waveform
        frame = waveform[start_idx:end_idx].copy()

        # Apply a Hamming window to the frame
        frame = frame * np.hamming(frame_size)

        # Compute the FFT of the frame
        spectrum = np.fft.fft(frame)

        # Take the real part of the spectrum
        spectrum = np.real(spectrum)

        # Store the spectrum in the spectrogram array
        spectrogram[frame_idx, :] = spectrum

    return spectrogram


def istft(spectrogram, frame_size=1024, frame_shift=256):
    """
    Cumpute the Inverse Short-Time Fourier Transform (ISTFT) of a spectrogram.

    Args:
        spectrogram (ndarray): Input spectrogram.
        frame_size (int): Size of each frame. Defaults to 1024.
        frame_shift (int): Number of samples to shift between frames. Defaults to 256.

    Returns:
        ndarray: Reconstructed waveform.
    """
    # Get the number of frames of the spectrogram
    num_frames = spectrogram.shape[0]

    # Calculate the number of samples and FFT size
    num_samples = (num_frames - 1) * frame_shift + frame_size
    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    # Make an empty array for the waveform
    waveform = np.zeros(num_samples)
    for frame_idx in range(num_frames):
        start_idx = frame_idx * frame_shift
        end_idx = start_idx + frame_size

        # Extract the current spectrum from the spectrogram
        spectrum = spectrogram[frame_idx, :]

        # Compute the inverse FFT of the spectrum
        frame = np.fft.ifft(spectrum)

        # Take the real part of the frame
        frame = np.real(frame)

        # Apply a Hamming window to the frame
        frame = frame * np.hamming(frame_size)

        # Add the frame to the waveform array
        waveform[start_idx:end_idx] += frame

    return waveform


def main():
    # Define the name of the wav file
    wav_name = 'ex1.wav'

    # Get waveform and sampling rate from the wav file
    wav, sr = sf.read(wav_name)

    # Compute the spectrogram using STFT
    spec = stft(wav)

    # Convert the spectrogram to decibels
    spec_db = 20 * np.log(np.abs(spec[:, :513]) + 1e-7)

    # Reconstruct the waveform from the spectrogram using ISTFT
    wav_hat = istft(spec)

    # Pad the reconstructed waveform to match the length of the original wavform
    wav_hat = np.pad(wav_hat, (0, wav.shape[0] - wav_hat.shape[0]), 'constant')

    # Create a figure to display three signals
    fig, axs = plt.subplots(3, 1, layout='constrained',
                            figsize=(10, 12), sharex=True)

    # Create a time axis for plotting
    time_axis = np.arange(wav.shape[0]) / sr

    # Plot the original signal
    pcm = axs[0].plot(time_axis, wav)
    axs[0].set_title('Original signal')
    axs[0].set_ylabel('Magnitude')
    axs[0].set_yticks([-0.2, 0.0, 0.2])

    # Plot the spectrogram
    # (It needs to be transposed and fliped upside-down)
    pcm = axs[1].imshow(np.flipud(spec_db.T),
                        extent=[0, wav.shape[0] / sr, 0, sr // 2000],
                        aspect='auto')
    axs[1].set_title('Spectrogram')
    axs[1].set_ylabel('Frequency [kHz]')
    axs[1].set_yticks(np.arange(0, 10, step=2))
    fig.colorbar(pcm, ax=axs[1], location='right')

    # Plot the re-synthesized signal
    pcm = axs[2].plot(time_axis, wav_hat)
    axs[2].set_title('Re-synthesized signal')
    axs[2].set_ylabel('Magnitude')
    axs[2].set_yticks([-0.2, 0.0, 0.2])

    # Set a common label for the x-axis
    fig.supxlabel('Time[s]')

    # Save the figure
    plt.savefig('signals.png')

    # Save the wav file (optional)
    sf.write('ex1_re.wav', wav_hat, sr)


if __name__ == "__main__":
    main()
