# Code for Ubicoustics-Android
A conversion of [Ubicoustics](https://github.com/FIGLAB/ubicoustics) for Android through Edge and Stream.


## Requirements
This is written in ```python3```.

Install as follows:
```
pip install -r requirements.txt
```

##  Python Usage

Use ```python3 add_preprocessing.py``` to generate a new ubicoustics model with preprocessing added within the network.

Use ```python3 quantize.py``` to convert the preprocessing model into a quantized ```tflite``` model.

Use ```python3 add_metadata.py``` to add ```tflite``` metadata for usage in Android. This is a separate file as ```tflite-support-nightly``` only has support for Linux and MacOS. Windows users can install the package and run the file through WSL.

Use ```python3 ubicoustics_stream.py``` to start the UDP streaming implementation of Ubicoustics. Make sure you set your IP.

All data analysis is found in ```data_analysis.ipynb```.

## Android Usage

All Android code is found in ```\Android```.

IP for stream can be set in ```\Android\app\src\main\java\com\example\ubicoustics_android_local\AudioStreamThread.kt```.

Please place the generated edge classifier in ```D:\Code\ubicoustics-conversion\Android\app\src\main\assets``` to run the edge classifier.

Upon building the app on your phone you will see two buttons for Edge or Stream. Edge is defaulted to saving all data onto a csv found in external storage with the path printed through LogCat. This can be disabled through companion variables in ```Android\app\src\main\java\com\example\ubicoustics_android_local\EdgeClassifier.kt```.
