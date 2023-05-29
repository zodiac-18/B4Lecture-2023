"""Main."""
import argparse

import librosa
import scipy.signal as signal

import envelope as env
import F0


def parse_args():
    """Retrieve variables from the command prompt."""
    parser = argparse.ArgumentParser(
        description="Generate spectrogram and inverse transform"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="input wav file"
    )
    parser.add_argument(
        "--nfft",
        type=int,
        default=1024,
        help="number of FFT points"
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="number of samples between successive STFT columns",
    )
    parser.add_argument(
        "--lifter",
        type=int,
        default=32,
        help="number of liftering",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=15,
        help="number of dimension",
    )
    parser.add_argument(
        "--window", type=str, default="hann", help="window function type"
    )
    return parser.parse_args()


def main():
    """Generate spectrograms and inverse transforms of audio signals."""
    args = parse_args()

    file_path = args.input_file
    # Download input file
    data, rate = librosa.load(file_path, mono=True)
    # Convert data to 1D
    data = data.flatten()

    # Parameter of stft
    nfft = args.nfft
    hop_length = args.hop_length
    lifter = args.lifter
    dimension = args.dimension
    window = args.window
    window_func = signal.get_window(window, nfft)

    # plot F0
    F0.plot_F0(data, rate, nfft, hop_length, lifter, window_func)

    # plot envelope
    env.plot_envelope(data, nfft, rate, lifter, dimension)


if __name__ == "__main__":
    main()
