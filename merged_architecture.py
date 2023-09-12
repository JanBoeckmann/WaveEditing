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

FILES_PATH = os.path.join(os.getcwd(), "Abschlussprojekt", "src")

samplerate, data = wavfile.read(os.path.join(FILES_PATH, "NoFX", "G61-40100-1111-20593.wav"))
length = data.shape[0] / samplerate

list_of_no_fx_files = os.listdir(os.path.join(FILES_PATH, "NoFX"))
list_of_overdrive_files = os.listdir(os.path.join(FILES_PATH, "Overdrive"))

overdrive_effect = '1'
new_sample_rate =  8000

input_data = list()
output_data = list()

scale = 19900

maximum = 0

for filename in list_of_no_fx_files:
    filtered_overdrive_file = [x for x in list_of_overdrive_files if x.startswith(filename[:9]) and x[13] == overdrive_effect]
    sampling_rate, data_no_fx = wavfile.read(os.path.join(FILES_PATH, "NoFX", filename))
    _, data_overdrive = wavfile.read(os.path.join(FILES_PATH, "Overdrive", filtered_overdrive_file[0]))
    number_of_samples = round(len(data) * float(new_sample_rate) / sampling_rate)
    data_no_fx = sps.resample(data_no_fx, number_of_samples)
    data_overdrive = sps.resample(data_overdrive, number_of_samples)
    if maximum < max(data_no_fx) or maximum < max(data_overdrive) or maximum < -min(data_no_fx) or maximum < -min(data_overdrive):
        maximum = max(max(data_no_fx), max(data_overdrive), -min(data_no_fx), -min(data_overdrive))
    data_no_fx = np.float32(data_no_fx/scale)
    data_overdrive = np.float32(data_overdrive/scale)
    input_data.append(data_no_fx)
    output_data.append(data_overdrive)

print("maximum unscaled value:", maximum)

input_data = np.array(input_data)
output_data = np.array(output_data)

x_train, x_test, y_train, y_test = train_test_split(
    input_data, output_data, test_size=0.3, random_state=0)

x_test, x_val, y_test, y_val = train_test_split(
    x_test, y_test, test_size=0.5, random_state=0)

# U-Net Hyperparameters
depth = 2
reduce_steps = 2
kernel_size = 5
number_channels = 32
reduce_factor = 2
use_bias = True
activation = "relu"

#Wave-Net Hyperparameters
diliations = (1, 2, 4, 8, 16, 32, 64)

out = []

reload_model = True

if not reload_model:
    #Build U-Net
    input_layer = keras.Input((2*new_sample_rate, 1))
    conv = keras.layers.Conv1D(number_channels, kernel_size, activation=activation, padding="same", use_bias=use_bias)(input_layer)
    for j in range(reduce_steps):
        for i in range(depth):
            conv = keras.layers.Conv1D(number_channels, kernel_size, activation=activation, padding="same", use_bias=use_bias)(conv)
        out.append(conv)
        conv = keras.layers.AveragePooling1D(reduce_factor)(conv)

    for j in range(reduce_steps):
        for i in range(depth):
            conv = keras.layers.Conv1D(number_channels, kernel_size, activation=activation, padding="same", use_bias=use_bias)(conv) 
        conv = keras.layers.UpSampling1D(reduce_factor)(conv)
        conv = keras.layers.Add()([conv, out[reduce_steps - 1 - j]])

    for i in range(depth):
        conv = keras.layers.Conv1D(number_channels, kernel_size, activation=activation, padding="same", use_bias=use_bias)(conv)
    output_layer_u_net = conv
    # output_layer_u_net = keras.layers.Dense(1, activation="tanh")(conv)

    # Wave Net
    out = []
    caus_conv = keras.layers.Conv1D(32, 3, padding="causal")(input_layer)
    for rate in diliations:
        dil_conv = keras.layers.Conv1D(32, 3, dilation_rate=rate, padding="causal")(caus_conv)
        act_tanh = keras.activations.tanh(dil_conv)
        act_sig = keras.activations.sigmoid(dil_conv)
        multiplied = keras.layers.Multiply()([act_tanh, act_sig])
        skip_conv = keras.layers.Conv1D(1, 1)(multiplied)
        out.append(skip_conv)
        caus_conv = keras.layers.Add()([caus_conv, skip_conv])
    add = keras.layers.Add()(out)
    act_relu = keras.layers.ReLU()(add)
    output_layer_wave_net = keras.layers.Conv1D(32,3, activation="ReLU", padding="same")(act_relu)
    #act_relu = keras.layers.ReLU()(conv)
    #conv = keras.layers.Conv1D(1,1)(act_relu)
    # output_layer_wave_net = keras.layers.Dense(1, activation="tanh")(conv)

    #combine Models
    combine = keras.layers.Concatenate()([output_layer_wave_net, output_layer_u_net])
    conv = keras.layers.Conv1D(32,3, activation="ReLU", padding="same")(combine)
    output_layer = keras.layers.Dense(1, activation="tanh")(conv)

    model = keras.Model(input_layer, output_layer)
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    early_stopping = keras.callbacks.EarlyStopping(patience=20, monitor="val_loss", mode="min", restore_best_weights=True)

    model.compile(loss='mean_squared_error', optimizer=opt)
    model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=[early_stopping], verbose=2, batch_size=16)
else:
    model = keras.models.load_model("merged_model" + str(depth) + "_" + str(reduce_steps) + "_" + str(kernel_size) + "_" + str(number_channels) + "_" + str(reduce_factor) + ".keras")
    model.summary()

if not reload_model:
    model.save("merged_model" + str(depth) + "_" + str(reduce_steps) + "_" + str(kernel_size) + "_" + str(number_channels) + "_" + str(reduce_factor) + ".keras")


start = timeit.timeit()
pred = model.predict(np.array([x_test[0]]))
end = timeit.timeit()

print("______________")
print("time for one forward pass:", end - start)
print("______________")

plot_pred = True
if plot_pred:
    time = np.linspace(0., length, 2*new_sample_rate)
    plt.plot(time, pred[0].reshape(-1, 1), label="Predicted Overdrive Signal")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

    time = np.linspace(0., length, 2*new_sample_rate)
    plt.plot(time, y_test[0].reshape(-1, 1), label="Original Overdrive Signal")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


y_test_pred = model.predict(y_test)
print("______________")
print("r2 between original signal and ground truth:", r2_score(x_test.reshape(-1, 1), y_test.reshape(-1, 1)))
print("r2 between predicted signal and ground truth:", r2_score(y_test_pred.reshape(-1, 1), y_test.reshape(-1, 1)))
print("______________")

output_scale = 100

test_pred = output_scale*pred[0].reshape(-1, 1)

old_number_of_samples = round(len(test_pred) * sampling_rate / float(new_sample_rate))
test_pred_rsp = sps.resample(test_pred, old_number_of_samples)

wavfile.write("example.wav", samplerate, test_pred_rsp)