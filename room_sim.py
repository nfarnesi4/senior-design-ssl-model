import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
import matplotlib.pyplot as plt

# specify dimensions for test room (in meters)
room_length = 10.0
room_width  = 10.0
room_height = 4.0

room_traverse_scale = 1 #change to 0.1

# audio threshold
# this value determines the first non-zero mag
# in a generated audio frame
audio_thresh = 10

# generate model for room object
# specify corners of room
corners = np.array([[0.0,0.0], [0.0,room_width], [room_length,room_width], [room_length,0.0]]).T  # [x,y]
#room = pra.Room.from_corners(corners)

# specify signal source
fs, signal = wavfile.read("sounds/white_noise96.wav")

window_size = 1024
signal = signal[0:window_size*4]
# set max_order to a low value for a quick (but less accurate) RIR
#room = pra.Room.from_corners(corners, fs=fs, max_order=8, absorption=0.2)
#room.extrude(room_height)

# add 8-microphone array
# specify 8-microphone array positions
R = np.array([[2.5, 2.5, 3.0, 3.0, 2.5, 2.5, 3.0, 3.0],\
              [2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0],\
              [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]])  # [[x], [y], [z]]
#room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

xs = np.arange(np.max(R[0])+0.5, room_length, room_traverse_scale)
ys = np.arange(np.max(R[1])+0.5, room_width,  room_traverse_scale)
zs = np.arange(np.max(R[2])+0.5, room_height, room_traverse_scale)

num_mic_channels = len(R[0])
total_frames = len(xs)*len(ys)*len(zs)
audio_data_length = window_size
audio_data = np.zeros((total_frames, num_mic_channels, audio_data_length))
pos_data = np.zeros((total_frames, 3))
frame = 0

print("Generating %d frames" % (total_frames))
print("image shape:", np.shape(audio_data))
print("pos shape:", np.shape(pos_data))

for z_coord in zs:
    for y_coord in ys:
        for x_coord in xs:

            # generate model for room object
            # set max_order to a low value for a quick (but less accurate) RIR
            room = pra.Room.from_corners(corners, fs=fs, max_order=8, absorption=0.5)
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


np.savez("dataset.npz", audio_data=audio_data, pos_data=pos_data)

plt.figure()
#plt.subplot(9,1,1)
plt.plot(signal)
plt.title('orignal signal')
for i in range(1):
    plt.figure()
    for mic,audio in enumerate(audio_data[i][:]):
        plt.subplot(8,1,mic+1)
        plt.plot(audio)
        plt.title(f'mic {mic}')
        plt.xlabel('sample')
        plt.ylabel('amp')

plt.show()
