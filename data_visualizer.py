import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

dataset = np.load("white_noise_dataset.npz")

audio_frames = dataset["audio_data"]
pos_frames = dataset["pos_data"]

fs, signal = wavfile.read("white_noise.wav")

signal = signal[0:2048]
print(len(signal))

#plt.figure()
#plt.plot(signal)
#plt.title('Orig')
#plt.xlim([0, len(signal)])
#plt.show()

for i in range(1):
    plt.figure()
    #plt.subplot(9,1,1)
    plt.plot(signal)
    plt.title('orignal signal')
    plt.figure()
    for mic,audio in enumerate(audio_frames[i][:]):
        plt.subplot(8,1,mic+1)
        plt.plot(audio)
        plt.title(f'mic {mic}')
        plt.xlabel('sample')
        plt.ylabel('amp')

#plt.legend([str(m) for m in range(len(audio_frames[0][:]))])
plt.show()
