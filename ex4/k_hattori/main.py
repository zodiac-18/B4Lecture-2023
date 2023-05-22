import numpy as np
import matplotlib.pyplot as plt
import myfunc as mf

def f0_autoco(data):
    autoco = np.correlate(data, data, mode="full")
    # for i in range(autoco.shape[0]-2):
    #     if autoco[i]<autoco[i+1] and autoco[i+1]>autoco[i+2]:
    plt.plot(autoco)
    plt.show()
    return 0

def cepstrum(data, samplerate):
    freq = np.linspace(0, samplerate, len(data))
    fft_data = np.fft.fft(data)
    abs_data = np.abs(fft_data[freq <= 8000])
    log_data = np.log(abs_data)

    ceps= np.fft.ifft(log_data)


    plt.plot(np.abs(ceps))
    plt.show()

    # ceps[10000:] = 0

    spec_envelope = np.fft.fft(ceps)

    plt.plot(ceps)
    plt.show()
    plt.plot(freq[freq <= 8000], log_data)
    plt.plot(freq[freq <= 8000], np.log(np.abs(spec_envelope)),"orange")
    plt.show()

    return 0

def main():
    data, samplerate = mf.wavload("ex4/k_hattori/ONSEI.wav")


    cep_data = cepstrum(data, samplerate)




if __name__ == "__main__":
    main()
    exit(1)