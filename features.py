# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Feature computation for YAMNet."""

import numpy as np
import tensorflow as tf
import sys

import mel_features

import vggish_params as params
# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):

  nyquist_hertz = audio_sample_rate / 2.
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
  spectrogram_bins_hertz = tf.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
  spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)

  band_edges_mel = tf.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
  # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
  # of spectrogram values.
  mel_weights_matrix = tf.reshape(tf.convert_to_tensor(()), (num_spectrogram_bins, num_mel_bins))
  for i in range(num_mel_bins):
    lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
    lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
    upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
    # .. then intersect them with each other and zero.
    mel_weights_matrix[:, i] = tf.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
  mel_weights_matrix[0, :] = 0.0
  return mel_weights_matrix

def waveform_to_log_mel_spectrogram_patches(waveform, params=params):
  """Compute log mel spectrogram patches of a 1-D waveform."""
  with tf.name_scope('log_mel_features'):
    # waveform has shape [<# samples>]

    # Convert waveform into spectrogram using a Short-Time Fourier Transform.
    # Note that tf.signal.stft() uses a periodic Hann window by default.
    window_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_WINDOW_LENGTH_SECONDS))
    hop_length_samples = int(
      round(params.SAMPLE_RATE * params.STFT_HOP_LENGTH_SECONDS))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    num_spectrogram_bins = fft_length // 2 + 1

    magnitude_spectrogram = _tflite_stft_magnitude(
        signal=waveform,
        frame_length=window_length_samples,
        frame_step=hop_length_samples,
        fft_length=fft_length)

    # magnitude_spectrogram = mel_features.stft_magnitude(
    #     signal=waveform,
    #     window_length=window_length_samples,
    #     hop_length=hop_length_samples,
    #     fft_length=fft_length
    #     )

    # Convert spectrogram into log mel spectrogram.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=params.NUM_MEL_BINS,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=params.SAMPLE_RATE,
        lower_edge_hertz=params.MEL_MIN_HZ,
        upper_edge_hertz=params.MEL_MAX_HZ)
    mel_spectrogram = tf.matmul(
      magnitude_spectrogram, linear_to_mel_weight_matrix)

    log_mel_spectrogram = tf.math.log(mel_spectrogram + params.LOG_OFFSET)
    # log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands]

    # Frame spectrogram (shape [<# STFT frames>, params.mel_bands]) into patches
    # (the input examples). Only complete frames are emitted, so if there is
    # less than params.patch_window_seconds of waveform then nothing is emitted
    # (to avoid this, zero-pad before processing).

    features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        params.EXAMPLE_HOP_SECONDS * features_sample_rate))

    features = tf.signal.frame(
        signal=log_mel_spectrogram,
        frame_length=example_window_length,
        frame_step=example_hop_length,
        axis=0)

    # features has shape [<# patches>, <# STFT frames in an patch>, params.mel_bands]
    return log_mel_spectrogram, features

def spectrogram_to_patches(log_mel_spectrogram, params):
  features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      params.EXAMPLE_HOP_SECONDS * features_sample_rate))

  features = tf.signal.frame(
      signal=log_mel_spectrogram,
      frame_length=example_window_length,
      frame_step=example_hop_length,
      axis=0)
  tf.print(features.shape)

  return features

def pad_waveform(waveform, params):
  """Pads waveform with silence if needed to get an integral number of patches."""
  # In order to produce one patch of log mel spectrogram input to YAMNet, we
  # need at least one patch window length of waveform plus enough extra samples
  # to complete the final STFT analysis window.
  min_waveform_seconds = (
      params.EXAMPLE_WINDOW_SECONDS +
      params.STFT_WINDOW_LENGTH_SECONDS - params.STFT_HOP_LENGTH_SECONDS)
  min_num_samples = tf.cast(min_waveform_seconds * params.SAMPLE_RATE, tf.int32)
  num_samples = tf.shape(waveform)[0]
  num_padding_samples = tf.maximum(0, min_num_samples - num_samples)

  # In addition, there might be enough waveform for one or more additional
  # patches formed by hopping forward. If there are more samples than one patch,
  # round up to an integral number of hops.
  num_samples = tf.maximum(num_samples, min_num_samples)
  num_samples_after_first_patch = num_samples - min_num_samples
  hop_samples = tf.cast(params.EXAMPLE_HOP_SECONDS * params.SAMPLE_RATE, tf.int32)
  num_hops_after_first_patch = tf.cast(tf.math.ceil(
          tf.cast(num_samples_after_first_patch, tf.float32) /
          tf.cast(hop_samples, tf.float32)), tf.int32)
  num_padding_samples += (
      hop_samples * num_hops_after_first_patch - num_samples_after_first_patch)

  padded_waveform = tf.pad(waveform, [[0, num_padding_samples]],
                           mode='CONSTANT', constant_values=0.0)
  return padded_waveform


def _tflite_stft_magnitude(signal, frame_length, frame_step, fft_length):
  """TF-Lite-compatible version of tf.abs(tf.signal.stft())."""
  def _hann_window():
    return tf.reshape(
      tf.constant(
        (0.5 - (0.5 * np.cos(2 * np.pi / frame_length * np.arange(frame_length)))
          ).astype(np.float32),
          name='hann_window'), [1, frame_length])

  def _dft_matrix(dft_length):
    """Calculate the full DFT matrix in NumPy."""
    # See https://en.wikipedia.org/wiki/DFT_matrix
    omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
    # Don't include 1/sqrt(N) scaling, tf.signal.rfft doesn't apply it.
    return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))

  def _rdft(framed_signal, fft_length):
    """Implement real-input Discrete Fourier Transform by matmul."""
    # We are right-multiplying by the DFT matrix, and we are keeping only the
    # first half ("positive frequencies").  So discard the second half of rows,
    # but transpose the array for right-multiplication.  The DFT matrix is
    # symmetric, so we could have done it more directly, but this reflects our
    # intention better.
    complex_dft_matrix_kept_values = _dft_matrix(fft_length)[:(
        fft_length // 2 + 1), :].transpose()
    real_dft_matrix = tf.constant(
        np.real(complex_dft_matrix_kept_values).astype(np.float32),
        name='real_dft_matrix')
    imag_dft_matrix = tf.constant(
        np.imag(complex_dft_matrix_kept_values).astype(np.float32),
        name='imaginary_dft_matrix')
    signal_frame_length = tf.shape(framed_signal)[-1]
    half_pad = (fft_length - signal_frame_length) // 2
    padded_frames = tf.pad(
        framed_signal,
        [
            # Don't add any padding in the frame dimension.
            [0, 0],
            # Pad before and after the signal within each frame.
            [half_pad, fft_length - signal_frame_length - half_pad]
        ],
        mode='CONSTANT',
        constant_values=0.0)
    real_stft = tf.matmul(padded_frames, real_dft_matrix)
    imag_stft = tf.matmul(padded_frames, imag_dft_matrix)
    return real_stft, imag_stft

  def _complex_abs(real, imag):
    return tf.sqrt(tf.add(real * real, imag * imag))

  framed_signal = tf.signal.frame(signal, frame_length, frame_step)
  windowed_signal = framed_signal * _hann_window()
  real_stft, imag_stft = _rdft(windowed_signal, fft_length)
  stft_magnitude = _complex_abs(real_stft, imag_stft)
  return stft_magnitude
