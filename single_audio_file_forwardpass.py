import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
import keras
from sklearn.preprocessing import MinMaxScaler
from unet import Unet
import scipy.signal as sps
from sklearn.metrics import r2_score
import timeit

filename = "G61-40100-1111-20593.wav"

FILES_PATH = os.path.join(os.getcwd(), "Abschlussprojekt")

new_sample_rate =  8000

input_data = list()
output_data = list()

scale = 19900

sampling_rate, data_no_fx = wavfile.read(os.path.join(FILES_PATH, filename))
length = data_no_fx.shape[0] / sampling_rate
number_of_samples = round(len(data_no_fx) * float(new_sample_rate) / sampling_rate)
data_no_fx = sps.resample(data_no_fx, number_of_samples)

#cut to two seconds
data_no_fx = data_no_fx[:2*new_sample_rate]

#apply scaling
data_no_fx = np.float32(data_no_fx/scale)

#put into array
input_data.append(data_no_fx)
input_data = np.array(input_data)

model = keras.models.load_model("merged_model2_2_5_32_2.keras")
model.summary()

pred = model.predict(input_data)

plot_pred = True
if plot_pred:
    time = np.linspace(0., length, 2*new_sample_rate)
    plt.plot(time, pred[0].reshape(-1, 1), label="Predicted Overdrive Signal")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

output_scale = 100

test_pred = output_scale*pred[0].reshape(-1, 1)

old_number_of_samples = round(len(test_pred) * sampling_rate / float(new_sample_rate))
test_pred_rsp = sps.resample(test_pred, old_number_of_samples)

wavfile.write("example2.wav", sampling_rate, test_pred_rsp)
