#!/usr/bin/env python

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
from scipy.io import loadmat
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft
import copy

def xcorr_freq(s1,s2):
    pad1 = np.zeros(len(s1))
    pad2 = np.zeros(len(s2))
    s1 = np.hstack([s1,pad1])
    s2 = np.hstack([pad2,s2])
    f_s1 = fft(s1)
    f_s2 = fft(s2)
    f_s2c = np.conj(f_s2)
    f_s = f_s1 * f_s2c
    denom = abs(f_s)
    denom[denom < 1e-6] = 1e-6
    f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation
    return np.abs(ifft(f_s))[1:]

dataset = np.load("dataset.npz")

audio_images = dataset["audio_data"]
responses = dataset["pos_data"]

#responses = responses / abs(responses).max()
frames, mics, window_size = audio_images.shape

mid = math.floor((2*window_size-1)/2)
mid2 = math.floor(mid/2)
mid_is = [i for i in range(mid-mid2,mid+mid2+2)]

for i, image in enumerate(audio_images):
    first = copy.deepcopy(image[0,:])
    for chan in range(mics):
        #cross = np.correlate(first,image[chan,:], mode='full')
        cross = xcorr_freq(first, image[chan,:])
        audio_images[i][chan] = cross[mid_is]
        #audio_images[i][chan][:] = cross[mid_is]


for i in range(1):
    plt.figure()
    for mic,audio in enumerate(audio_images[i][:]):
        plt.subplot(8,1,mic+1)
        plt.plot(audio)
        plt.title(f'mic {mic}')
        plt.xlabel('sample')
        plt.ylabel('amp')

#plt.show()

audio_images = audio_images.reshape([-1,mics,window_size,1])

#dataset = tf.data.Dataset.from_tensor_slices((audio_images, responses))
#dataset = dataset.shuffle(len(audio_images))


#for feat, targ in dataset.take(5):
  #print ('Features: {}, Target: {}'.format(str(feat.shape), targ))

#dataset.batch(batch_size)

#layers = [
    #imageInputLayer([height width channels])

    #convolution2dLayer([1 200], 4, 'padding', 'same')
    #batchNormalizationLayer
    #convolution2dLayer([8 40], 6, 'padding', 'same')
    #fullyConnectedLayer(2*window_size)
    #%fullyConnectedLayer(window_size)
    #% output later
    #dropoutLayer(0.5)
    #fullyConnectedLayer(3)
    #regressionLayer
    #];

model = Sequential()
model.add(Conv2D(4, (1, 200)))
model.add(Conv2D(6, (8, 40)))
model.add(Flatten())
model.add(Dense(window_size*2))
model.add(Dense(window_size))
#model.add(Dropout(0.2))
model.add(Dense(3))

print(model.summary)
model.compile(loss='mse',
              optimizer=keras.optimizers.Adadelta(learning_rate=0.25),
              metrics=['mae'])

#plot_model(model, to_file='model.png')

model.fit(x=audio_images,
          y=responses,
          epochs=5,
          batch_size=16,
          shuffle=True,
          validation_split=0.1,
          verbose=1)

dataset = np.load("spiral.npz")

audio_images = dataset["audio_data"]
responses = dataset["pos_data"]

#responses = responses / abs(responses).max()
frames, mics, window_size = audio_images.shape

mid = math.floor((2*window_size-1)/2)
mid2 = math.floor(mid/2)
mid_is = [i for i in range(mid-mid2,mid+mid2+2)]

for i, image in enumerate(audio_images):
    first = copy.deepcopy(image[0,:])
    for chan in range(mics):
        #cross = np.correlate(first,image[chan,:], mode='full')
        cross = xcorr_freq(first, image[chan,:])
        audio_images[i][chan] = cross[mid_is]
        #audio_images[i][chan][:] = cross[mid_is]

audio_images = audio_images.reshape([-1,mics,window_size,1])
responses_p = model.predict(audio_images)
print(responses_p.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(responses[:,0], responses[:,1], responses[:,2])
ax.scatter(responses_p[:,0], responses_p[:,1], responses_p[:,2])
plt.show()
