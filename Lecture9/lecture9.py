import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import random
from scipy.io import wavfile



frequency_sampling, audio_signal = wavfile.read('file.wav')

print('\n Signal shape', audio_signal.shape)
print('\n Signal data type', audio_signal.dtype)
print('\n Signal duration', round(audio_signal.shape[0]/float(frequency_sampling),2))

audio_signal = audio_signal/np.power(2,15)
signal = audio_signal[0:100]

time_axis = 1000*np.arange(0, len(signal), 1)/float(frequency_sampling)

plt.plot(time_axis, signal, color='red')
plt.xlabel('Time (miliseconds)')
plt.ylabel('Amplitude')
plt.title('Input Audio File')
plt.show()


duration = 4 
frequency_sampling_2 =  44100
frequency_tone = 784

min_val= -4*np.pi
max_val = 4*np.pi

t = np.linspace(min_val,max_val, duration*frequency_sampling_2)
audio_signal = np.sin(2*np.pi*frequency_tone*t)
output_file = 'generated_file.wav'

write(output_file, frequency_sampling_2, audio_signal)

signal = audio_signal[:100]
time_axis = 1000*np.arange(0, len(signal), 1)/float(frequency_sampling_2)

plt.plot(time_axis, signal, color='red')
plt.xlabel('Time (miliseconds)')
plt.ylabel('Amplitude')
plt.title('Input Audio File')
plt.show()


rand1 = np.random.uniform(-1.0, 1.0, size=len(signal))

signal = signal + rand1
plt.plot(time_axis, signal, color='red')
plt.xlabel('Time (miliseconds)')
plt.ylabel('Amplitude')
plt.title('Input Audio File')
plt.show()