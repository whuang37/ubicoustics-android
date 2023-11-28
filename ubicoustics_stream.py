from keras.models import load_model
import numpy as np
from vggish_input import waveform_to_examples
import ubicoustics
from pathlib import Path
import time
import wget

import socket
import struct

MODEL_URL = "https://www.dropbox.com/s/cq1d7uqg0l28211/example_model.hdf5?dl=1"
MODEL_PATH = "models/example_model.hdf5"

CHANNELS = 1
RATE = 16000
CHUNK = RATE

UDP_IP = "0.0.0.0"  # Use "0.0.0.0" to bind to all available interfaces or just use the same one as in the server
UDP_PORT = 12345  # Make sure this matches the port used in the Kotlin client

###########################
# Download model, if it doesn't exist
###########################
def get_model(model_filename= "models/example_model.hdf5"):
    print("=====")
    print("1 / 1: Checking model... ")
    print("=====")
    ubicoustics_model = Path(model_filename)
    if (not ubicoustics_model.is_file()):
        print("Downloading example_model.hdf5 [867MB]: ")
        wget.download(MODEL_URL,MODEL_PATH)

    ##############################
    # Load Deep Learning Model
    ##############################
    print("Using deep learning model: %s" % (model_filename))
    model = load_model(model_filename)
    context = ubicoustics.everything

    label = dict()
    for k in range(len(context)):
        label[k] = context[k]

    return model, label


##############################
# Setup Audio
##############################
def audio_samples(in_data, model, label):
    np_wav = np.frombuffer(in_data, dtype=np.int16) / 32768.0 # Convert to [-1.0, +1.0]
    x = waveform_to_examples(np_wav, RATE)
    predictions = []
    if x.shape[0] != 0:
        x = x.reshape(len(x), 96, 64, 1)
        pred = model.predict(x)
        predictions.append(pred)

    p = ("", "")
    for prediction in predictions:
        m = np.argmax(prediction[0])
        if (m < len(label)):
            p = (label[m], prediction[0,m])
            print(p)
            print("Prediction: %s (%0.2f)" % p)
        else:
            print("KeyError: %s" % m)

    return p


def main():
    model, label = get_model()

    # Setup UDP
    initial_message_received = False

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print("Started Server")

    data_file_path = f"data/stream/{int(time.time()*1000)}.csv"

    with open(data_file_path, "w") as f:
        f.write("timestamp,label,probability,delay,record_delay,process_delay\n")

    while True:
        data, addr = sock.recvfrom(32016)  # Adjust the buffer size as needed
        if not initial_message_received:
            initial_message_received = True
            continue

        # Extract prev runs delay (assuming it's at the beginning of the packet)
        delay_bytes = data[:8]  # Assuming a long (8 bytes) timestamp
        record_delay = struct.unpack("!Q", delay_bytes)[0]

        # Extract audio data
        audio_data = data[8:]

        time_received = int(time.time()*1000)
        predicted_label, probability = audio_samples(audio_data, model, label)
        time_processed = int(time.time()*1000)

        process_delay = time_processed - time_received
        delay = record_delay + process_delay
        # Process the received data as needed
        # print(f"Received {len(audio_data)} bytes of audio data from {addr}")

        # write to log file
        with open(data_file_path, "a") as f:
            f.write(f"{time_received},{predicted_label},{probability},{delay},{record_delay},{process_delay}\n")

if __name__ == "__main__":
    main()
