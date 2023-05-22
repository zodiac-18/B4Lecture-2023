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
    fft_data = np.fft.fft(data)
    cept_data = np.fft.fft(fft_data)
    cept_data[150000:] = 0


    plt.subplot(211)
    plt.plot(fft_data)
    plt.subplot(212)
    plt.plot(cept_data)
    plt.show()
    plt.plot(20*np.log10(np.abs(fft_data)))
    plt.plot(20*np.log10(cept_data), "orange")
    plt.show()

    return 0

def main():
    data, samplerate = mf.wavload("ex4/k_hattori/ONSEI.wav")

    cep_data = cepstrum(data, samplerate)
    plt.plot(data)
    plt.plot(cep_data, "orange")
    plt.show()




if __name__ == "__main__":
    main()
    exit(1)