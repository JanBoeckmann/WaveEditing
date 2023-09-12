# https://www.idmt.fraunhofer.de/en/publications/datasets/audio_effects.html
"""
Created on Tue Nov 16 14:06:06 2021

@author: Bastian Kanning, Jan Boeckmann
"""

import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

FILES_PATH = os.path.join(os.getcwd(), "Abschlussprojekt", "src")

samplerate, data = wavfile.read(os.path.join(FILES_PATH, "NoFX", "G61-40100-1111-20593.wav"))
length = data.shape[0] / samplerate

time = np.linspace(0., length, data.shape[0])
plt.plot(time, data, label="Signal")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

plt.specgram(data)
plt.show()