from keras.models import load_model
import numpy as np
from vggish_input import waveform_to_examples
import ubicoustics
import pyaudio
from pathlib import Path
import time
import argparse
import wget

import features as features_lib

import tensorflow as tf
from tensorflow.keras import Model, layers

import vggish_params as params

# Variables
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE
MICROPHONES_DESCRIPTION = []
FPS = 60.0

def preprocessing_model(params):
    waveform = layers.Input(batch_shape=(params.SAMPLE_RATE,), dtype=tf.float32)

    window_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_WINDOW_LENGTH_SECONDS))
    hop_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_HOP_LENGTH_SECONDS))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    num_spectrogram_bins = fft_length // 2 + 1

    spectrogram = features_lib._tflite_stft_magnitude(
        signal=waveform,
        frame_length=window_length_samples,
        frame_step=hop_length_samples,
        fft_length=fft_length)

    # Convert spectrogram into log mel spectrogram.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=params.NUM_MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=params.SAMPLE_RATE,
        lower_edge_hertz=params.MEL_MIN_HZ,
        upper_edge_hertz=params.MEL_MAX_HZ)
    mel_spectrogram = tf.matmul(
      spectrogram, linear_to_mel_weight_matrix)

    log_mel_spectrogram = tf.math.log(mel_spectrogram + params.LOG_OFFSET)

    features = features_lib.spectrogram_to_patches(log_mel_spectrogram, params)

    preprocessing_included_model = Model(name="preprocessing",inputs=waveform, outputs=features)

    return preprocessing_included_model

def ubicoustics_preprocessing(m, params):
    pp_model = preprocessing_model(params)
    output = m(pp_model.output)

    full_model = Model(pp_model.input, output, name="full_processing")

    return full_model

if __name__ == "__main__":
    MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
    MODEL_PATH = "models/example_model.hdf5"
    SAVE_PATH = 'models/example_model_preprocess.hdf5'

    print("=====")
    print("2 / 2: Checking model... ")
    print("=====")
    model_filename = "models/example_model.hdf5"
    ubicoustics_model = Path(model_filename)
    if (not ubicoustics_model.is_file()):
        print("Downloading example_model.hdf5 [867MB]: ")
        wget.download(MODEL_URL,MODEL_PATH)

    ##############################
    # Load Deep Learning Model
    ##############################
    print("Using deep learning model: %s" % (model_filename))
    model = load_model(model_filename)

    preprocessing_model = ubicoustics_preprocessing(model, params)

    tf.keras.saving.save_model(preprocessing_model, SAVE_PATH, save_format="h5")
