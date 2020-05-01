import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy
from math import pi
from numpy import cos, sin


# audio threshold
# this value determines the first non-zero mag
# in a generated audio frame
audio_thresh = 10

# generate model for room object
# specify corners of room
corners = np.array([[0.0000,0.9906], [0.8128,0.9906], [0.8128,0.0000], [4.7244,0.0000],\
                    [4.7244,0.8382], [5.4864,0.8382], [5.4864,0.2540], [9.6520,0.2540],\
                    [9.6520,3.5306], [10.4648,3.5306], [10.4648,6.4516], [9.6520,6.4516],\
                    [9.6520,7.2390], [5.4864,7.2390], [5.4864,6.5532], [4.7244,6.5532],\
                    [4.7244,7.2390], [0.8890,7.2390], [0.8890,6.2738], [-0.0508,6.2738]]).T  # [x,y]
room_height = 2.3876

# specify signal source
fs, signal = wavfile.read("sounds/white_noise96.wav")

window_size = 1024
signal = signal[0:window_size*4]

# specify 8-microphone array positions
#R = np.array([[2.5, 2.5, 3.0, 3.0, 2.5, 2.5, 3.0, 3.0],\
#              [2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0],\
#             [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]])  # [[x], [y], [z]]

# mic array in the middle of the room
#R = np.array([[5.7324, 5.7324, 5.7324, 5.7324, 4.7324, 4.7324, 4.7324, 4.7324],\
#             [3.1195, 4.1195, 3.1195, 4.1195, 3.1195, 4.1195, 3.1195, 4.1195],\
#             [1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000]])  # [[x], [y], [z]]

# mic array with variable x,y
fl_x, fl_y = 1.5, 3.1195
array_size = 1
R = np.array([[fl_x+array_size, fl_x+array_size, fl_x+array_size, fl_x+array_size, fl_x  ,  fl_x           , fl_x  , fl_x           ],\
              [fl_y           , fl_y+array_size, fl_y           , fl_y+array_size, fl_y  ,  fl_y+array_size, fl_y  , fl_y+array_size],\
              [1.0000         , 1.0000         , 0.0000         , 0.0000         , 1.0000,  1.0000         , 0.0000, 0.0000         ]])  # [[x], [y], [z]]

total_time_s = 60
fps = 60
t = np.linspace(0, total_time_s-1,total_time_s*fps)

r = 2.5
center_x, center_y = 5.7, 3.7
z_speed = 2/total_time_s
omega = (2*pi)/total_time_s

x = r * cos(omega*t) + center_x
y = r * sin(omega*t) + center_y
z = z_speed * t

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

plt.figure()
plt.title("Bossone 302 Simulation: Top Down View")
plt.plot(np.append(corners[:][0], corners[0][0]),
         np.append(corners[:][1], corners[0][1]), '-o')
plt.scatter(R[:][0], R[:][1], color='green')
plt.plot([3, 3, 8.4, 8.4, 3], [1, 6.4, 6.4, 1, 1], '-o')
plt.scatter(x,y)
plt.legend(["floor plan", 'training set boarder',
            "mic array base", "test spiral"])

plt.show()
num_mic_channels = len(R[0])
total_frames = len(t)
audio_data_length = window_size
audio_data = np.zeros((total_frames, num_mic_channels, audio_data_length))
pos_data = np.zeros((total_frames, 3))
frame = 0

print("Generating %d frames" % (total_frames))
print("image shape:", np.shape(audio_data))
print("pos shape:", np.shape(pos_data))

for ts in range(len(t)):
    x_coord, y_coord, z_coord = x[ts], y[ts], z[ts]

    # generate model for room object
    # set max_order to a low value for a quick (but less accurate) RIR
    room = pra.Room.from_corners(corners, fs=fs, max_order=1, absorption=0.99)
    room.extrude(room_height)

    # add 8-microphone array
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    # add source and set the signal to WAV file content
    room.add_source([x_coord, y_coord, z_coord], signal=signal)

    # generate RIR and output audio for each microphone
    room.simulate()

    if frame % 100 == 0:
        print("Frame: %d" % (frame))

    non_zero_starts = []
    sizes = []
    for channel in range(num_mic_channels):
        sizes.append(len(room.mic_array.signals[channel,:]))
        for i,sample in enumerate(room.mic_array.signals[channel]):
            if abs(sample) > audio_thresh:
                non_zero_starts.append(i)
                break

    samp_offset = max(non_zero_starts)
    # store audio data in 2D numpy array
    #mic_channels = np.zeros((audio_data_length,num_mic_channels))
    for channel in range(num_mic_channels):
        #for sample in range(audio_data_length):
        for s in range(audio_data_length):
            audio_data[frame][channel][s] = room.mic_array.signals[channel,s+samp_offset]

    #audio_data[frame] = mic_channels
    pos_data[frame] = np.array([x_coord, y_coord, z_coord])
    frame += 1

    # store audio data in files with naming format L_W_H
    #filename = "./audio_data/{}_{}_{}.csv".format(x_coord, y_coord, z_coord)
    #np.savetxt(filename, mic_channels, delimiter=',')


np.savez("spiral_one.npz", audio_data=audio_data, pos_data=pos_data)

plt.figure()
#plt.subplot(9,1,1)
plt.plot(signal)
plt.title('orignal signal')
for i in range(1):
    plt.figure()
    plt.title('Raw Audio')
    for mic,audio in enumerate(audio_data[i][:]):
        plt.subplot(8,1,mic+1)
        plt.plot(audio)
        plt.title(f'mic {mic}')
        plt.xlabel('sample')
        plt.ylabel('amp')

    plt.figure()
    plt.title('CrossCorr Audio')
    first = copy.deepcopy(audio_data[i][0])
    for mic,audio in enumerate(audio_data[i][:]):
        plt.subplot(8,1,mic+1)
        cross = np.correlate(first,audio, mode='full')
        plt.plot(cross)
        plt.title(f'mic {mic}')
        plt.xlabel('sample')
        plt.ylabel('amp')

plt.show()
