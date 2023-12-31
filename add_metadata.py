from tflite_support.metadata_writers import audio_classifier
from tflite_support.metadata_writers import metadata_info
from tflite_support.metadata_writers import writer_utils


SAVE_PATH = "models/example_model_quantized.tflite"
SAVE_META_PATH = "models/example_model_quantized_meta.tflite"

AudioClassifierWriter = audio_classifier.MetadataWriter
_MODEL_PATH = SAVE_PATH
# Task Library expects label files that are in the same format as the one below.
_LABEL_FILE = "all_labels.txt"
# Expected sampling rate of the input audio buffer.
_SAMPLE_RATE = 16000
# Expected number of channels of the input audio buffer. Note, Task library only
# support single channel so far.
_CHANNELS = 1
_SAVE_TO_PATH = SAVE_META_PATH

# Create the metadata writer.
writer = AudioClassifierWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), _SAMPLE_RATE, _CHANNELS, [_LABEL_FILE])

# Verify the metadata generated by metadata writer.
print(writer.get_metadata_json())

# Populate the metadata into the model.
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)
