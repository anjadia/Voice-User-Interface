import numpy.fft
import numpy as np
from pathlib import Path

import scipy
import scipy.signal
import scipy.fftpack
from scipy.io import wavfile

import librosa
from ruamel.yaml import YAML

class MFCC():

  def __init__(self, config_file, audio_preprocessor):
    self.config_file = config_file
    self.preprocessor = audio_preprocessor

    if "MFCC" in config_file.keys():
      config_file = self.config_file["MFCC"]
      print("******\nLoaded MFCC config file.\n******")
      self.frame_len = config_file["frame_len"] if "frame_len" in config_file.keys() else 0.04
      self.hop_len = config_file["hop_len"] if "hop_len" in config_file.keys() else 0.01
      self.FFT_size = config_file["FFT_size"] if "FFT_size" in config_file.keys() else 1024
      self.window = config_file["window"] if "window" in config_file.keys() else "hann"
      self.n_mfcc = config_file["n_mfcc"] if "n_mfcc" in config_file.keys() else 30
      self.dct_type = config_file["dct_type"] if "dct_type" in config_file.keys() else 2
      self.dct_norm = config_file["dct_norm"] if "dct_norm" in config_file.keys() else "ortho"
    else:                
      print("Incorrect config file")
      raise 


  def compute_mfcc(self, signal, fs):
    """
    On given signal returns computed MFCC
    """
    framed_signal = self.frame_signal(signal, fs)

    fft_frames = self.stft(framed_signal)

    filters = self.get_filters(fs)

    mel_stft = np.dot(filters, fft_frames)

    log_mel_stft = self.take_log(mel_stft)

    computed_mfcc = self.DCT(log_mel_stft)

    return computed_mfcc
  

  def compute_mfcc(self, signal_path):
    """
    On given path load signal and returns computed MFCC
    """
    fs, signal = wavfile.read(signal_path)
    if len(signal.shape) == 2:
      signal = signal[:,1]

    signal, fs = self.preprocessor.audio_peprocessing(signal, fs)

    framed_signal = self.frame_signal(signal, fs)

    fft_frames = self.stft(framed_signal)

    filters = self.get_filters(fs)

    mel_stft = np.dot(filters, fft_frames)

    log_mel_stft = self.take_log(mel_stft)

    computed_mfcc = self.DCT(log_mel_stft)

    ##TO REMOVE##
    # librosa_mfcc = librosa.feature.mfcc(S = log_mel_stft, n_mfcc=30)
    # assert(np.allclose(librosa_mfcc, computed_mfcc, atol=1))
    ##---------##

    computed_mfcc = np.nan_to_num(computed_mfcc)

    return computed_mfcc


  def take_log(self, mel_stft):
    """ Compute log for given variable """ 
    return np.log(mel_stft)


  def frame_signal(self, signal, fs):
    """
    Split signal into frames, pad it with zeros to FFT_size
    """
    framed_signal = []
    frame_len = int(self.frame_len * fs)
    hop_len = int(self.hop_len * fs)
    begin = 0

    FFT_size = self.FFT_size * 2
    while begin + frame_len < len(signal):
      end = begin + frame_len

      frame = signal[begin:end]

      if len(frame) < FFT_size: 
        frame = np.concatenate([np.array(frame), np.zeros(FFT_size - len(frame))])

      framed_signal.append(frame)
      begin = begin + hop_len
      
      
    frame = signal[begin:]

    if len(frame) < FFT_size: 
      frame = np.concatenate([np.array(frame), np.zeros(FFT_size - len(frame))])

    framed_signal.append(frame)

    return framed_signal


  def stft(self, frames):
    """
    Compute fft for all given frames (STFT)
    """
    fft_frames = np.array([])
    fft_window = scipy.signal.get_window(self.window, self.FFT_size*2, fftbins=True)

    for frame in frames:

      if len(fft_frames) == 0:
        fft_frames = numpy.fft.rfft(fft_window * frame, n = self.FFT_size*2-1)

      else:
        fft_frames = np.vstack(
          [
            fft_frames,
            numpy.fft.rfft(fft_window * frame, n = self.FFT_size*2-1)
          ] 
        )
    return np.abs(fft_frames.T)**2 * 1/self.FFT_size
          

  def mel_to_frequency(self, mel):
    """
    Given a frequency value, it returns the equivalent value on the mel scale
    """
    return 700 *(np.exp(mel/1125 ) - 1)


  def frequency_to_mel(self, frequency):
    """
    Given a mel value, it returns the equivalent value on the frequency scale
    """
    return 1125 * np.log(1+frequency/700)


  def get_filter_points(self, fs):
    """
    Returns filter points to mel filters
    """
    fmin_mel = self.frequency_to_mel(0)
    fmax_mel = self.frequency_to_mel(fs / 2)
    
    FFT_size = self.FFT_size * 2 -1
    # print("MEL min: {0}".format(fmin_mel))
    # print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=self.n_mfcc+2)
    # print(mels)
    freqs = self.mel_to_frequency(mels)
    # print(freqs)
    # print(np.floor((self.FFT_size + 1) * freqs / fs ).astype(int))
    return np.floor((FFT_size + 1) * freqs / fs ).astype(int), freqs


  def get_filters(self, fs):
    """
    Returns n_mfcc number filters for fiven fs
    """
    FFT_size = self.FFT_size * 2 -1

    filter_points, mel_freqs = self.get_filter_points(fs=fs)
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(self.n_mfcc):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    enorm = 2.0 / (mel_freqs[2:self.n_mfcc+2] - mel_freqs[:self.n_mfcc])
    filters *= enorm[:, np.newaxis]

    return filters


  def DCT(self, log_mel_stft):
    """
    Compute DCT for given stft
    """
    try:
      return scipy.fftpack.dct(log_mel_stft, axis=-2, type=self.dct_type, norm=self.dct_norm)[:self.n_mfcc, :]
    except:
      return [[]]
    # return scipy.fftpack.dct(log_mel_stft, axis=-2, type=self.dct_type, norm=self.dct_norm)[..., :self.n_mfcc, :]


# if __name__ == "__main__":
#     import sys
#     import time

#     wav = sys.argv[1]

#     config_file = Path(__file__).absolute().parents[2] / "config.yaml"

#     with config_file.open() as f:
#         config = YAML(typ="safe").load(f)

#         mfcc_preprocessor = MFCC(config["asr"])

#         start = time.time()

#         audio_mfcc = mfcc_preprocessor.compute_mfcc(signal_path = wav) 
#         audio_mfcc = np.concatenate([audio_mfcc, audio_mfcc], axis = 1)
#         print(audio_mfcc.shape)

#         end = time.time()

#         print("Processed in %1.2f sec" % (end - start))