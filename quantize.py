from vggish_input import waveform_to_examples, wavfile_to_examples
import numpy as np
import tensorflow as tf
from keras.models import load_model
import vggish_params
from pathlib import Path
import ubicoustics
import wget

MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model_preprocess.hdf5"
SAVE_PATH = "models/example_model_quantized.tflite"


###########################
# Load Model
###########################
trained_model = MODEL_PATH

print("Using deep learning model: %s" % (trained_model))
model = load_model(trained_model)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open(SAVE_PATH, "wb") as f:
    f.write(tflite_quant_model)
