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
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time

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

model = keras.models.load_model('ssl_model')

dataset = np.load("spiral_one.npz")

audio_images = dataset["audio_data"]
responses = dataset["pos_data"]

#responses = responses / abs(responses).max()
frames, mics, window_size = audio_images.shape

mid = math.floor((2*window_size-1)/2)
mid2 = math.floor(mid/2)
mid_is = [i for i in range(mid-mid2,mid+mid2+2)]

start_time = time.time()
for i, image in enumerate(audio_images):
    first = copy.deepcopy(image[0,:])
    for chan in range(mics):
        #cross = np.correlate(first,image[chan,:], mode='full')
        cross = xcorr_freq(first, image[chan,:])
        audio_images[i][chan] = cross[mid_is]
        #audio_images[i][chan][:] = cross[mid_is]

audio_images = audio_images.reshape([-1,mics,window_size,1])
responses_p = model.predict(audio_images)
total_time = time.time() - start_time
print("Average inference time: %f ms" % (1000*total_time/len(responses_p)))
print(responses_p.shape)

errors = abs(responses-responses_p)
avg_error = sum(errors)/len(errors)
print(avg_error)

def update(num, data, scatters):
    res, rep = data
    scatters[0][0].set_data_3d(res[:num,0], res[:num,1], res[:num,2])
    scatters[1]._offsets3d = ([res[num,0]], [res[num,1]], [res[num,2]])
    scatters[2]._offsets3d = ([rep[num,0]], [rep[num,1]], [rep[num,2]])
    return scatters

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(responses[:,0], responses[:,1], responses[:,2])
#ax.scatter(responses_p[:,0], responses_p[:,1], responses_p[:,2])
plt.title('SSL Tracking')
plt.legend(['sound source path', 'sound source', 'estimated'])
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_xlim3d([0.0, 10.0])
ax.set_ylim3d([0.0, 10.0])
ax.set_zlim3d([-0.5, 2.5])
ax.view_init(10,10)

scatters = [ ax.plot([responses_p[0][0]], [responses_p[0][1]], [responses_p[0][2]]),
             ax.scatter([responses_p[0][0]], [responses_p[0][1]], [responses_p[0][2]]),
             ax.scatter([responses_p[0][0]], [responses_p[0][1]], [responses_p[0][2]])]

data=(responses, responses_p)
# Creating the Animation object
ani = animation.FuncAnimation(fig, update, len(responses), fargs=(data, scatters),
                              interval=1000/60, blit=False)

plt.legend(['sound source path', 'sound source', 'estimated'])

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60)


plt.show()

ani.save('spiral.mp4', writer=writer)
