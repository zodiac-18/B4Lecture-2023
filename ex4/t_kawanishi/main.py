"""f0 estimation and spectrum envelope."""
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np

import ex1Function as ex1
import f0_spe_Function as f0

if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="""This is a program to estimate sound's
                    Fundamental frequency and spectral envelop."""
    )
    parser.add_argument("path", help="path of the sound")
    parser.add_argument(
        "-s",
        "--f_size",
        help="frame size recommend between 128 to 2048 ",
        default=512,
        type=int,
    )
    parser.add_argument(
        "-r", "--overlap_r", help="overlap rate between 0 to 1", default=0.5, type=float
    )
    parser.add_argument(
        "-l",
        "--y_limit",
        help="limit of spectrogram's frequency",
        default=20000,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--degree",
        help="degree of the lpc",
        default=5,
        type=int,
    )
    # get arguments
    args = parser.parse_args()

    # get sound file
    data, s_rate = ex1.load_sound(args.path)

    # adapt to Autocorrelation
    f0_g1 = f0.f0_ACF(data, s_rate, size=args.f_size, overlap_r=args.overlap_r)

    # adapt to spectrum
    f0_g2 = f0.f0_cep(data, s_rate, lif=50, size=args.f_size, overlap_r=args.overlap_r)

    # create figure and sub figure
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # create spectrum
    ex1.show_spectrogram(
        data,
        ax1,
        Fs=args.f_size,
        overlap_r=args.overlap_r,
        sample_rate=s_rate,
        y_lim=args.y_limit,
    )
    ex1.show_spectrogram(
        data,
        ax2,
        Fs=args.f_size,
        overlap_r=args.overlap_r,
        sample_rate=s_rate,
        y_lim=args.y_limit,
    )

    # create file name
    name = re.sub(r".+\\", "", args.path)
    name = re.sub(r"\..+", "", name)

    # plot f0 generate by autocorrelation Function
    dis = int(args.f_size * args.overlap_r)
    frame_l = int(len(data) // dis) * dis
    x_t = np.arange(0, frame_l, dis)
    x_t = x_t / s_rate
    ax1.plot(x_t, f0_g1)
    ax1.set_title("spectrum and f0 estimation with autocorrelation")

    # plot f0 generate by cepstrum method
    ax2.plot(x_t, f0_g2)
    ax2.set_title("spectrum and f0 estimation with cepstrum")
    plt.savefig("f0_" + name)

    # spectrum envelop
    win = np.hamming(args.f_size)
    win_data = data[: args.f_size] * win

    log = 20 * np.log10(np.abs(np.fft.rfft(win_data)))
    cep = f0.cep_env(win_data, lif=50)
    lpc = f0.lpc(win_data, args.degree, args.f_size)

    plt.figure()
    f_group = np.fft.rfftfreq(args.f_size)
    plt.plot(f_group, log, label="Spectrum")
    plt.plot(f_group, cep[: len(log)], label="Cepstrum")
    plt.plot(f_group, lpc[: len(log)], label="LPC")
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Log amplitude spectrum[dB]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectrum_" + name + "_" + str(args.degree))
    plt.show()
