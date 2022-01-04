import numpy as np

import librosa

def mel_to_frequency(mel) -> np.float:
  """
  Given a frequency value, it returns the equivalent value on the mel scale
  """
  return 700 *(np.exp(mel/1125 ) - 1)


def frequency_to_mel(frequency) -> np.float:
  """
  Given a mel value, it returns the equivalent value on the frequency scale
  """
  return 1125 * np.log(1+frequency/700)


def get_filter_points(mel_filter_num, FFT_size, sample_rate):
  fmin_mel = frequency_to_mel(0)
  fmax_mel = frequency_to_mel(sample_rate / 2)
  
  print("MEL min: {0}".format(fmin_mel))
  print("MEL max: {0}".format(fmax_mel))
  
  mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
  print(mels)
  freqs = mel_to_frequency(mels)
  print(freqs)
  print(np.floor((FFT_size + 1) * freqs / sample_rate ).astype(int))
  return np.floor((FFT_size + 1) * freqs / sample_rate ).astype(int), freqs


def get_filters(sample_rate, FFT_size, mel_filter_num):

  filter_points, mel_freqs = get_filter_points(mel_filter_num, FFT_size, sample_rate=sample_rate)
  filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
  
  for n in range(mel_filter_num):
      filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
      filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
  
  enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
  filters *= enorm[:, np.newaxis]

  return filters


def mfcc(signal, fs, config_file):

    n_mfcc = config_file["n_mfcc"] if "n_mfcc" in config_file.keys() else 64
    n_fft = config_file["n_fft"] if "n_fft" in config_file.keys() else 2048
    hop_length = config_file["hop_length"] if "hop_length" in config_file.keys() else 0.01
    win_length = config_file["win_length"] if "win_length" in config_file.keys() else  0.02
    window = config_file["window"] if "window" in config_file.keys() else  "hann"
    center = config_file["center"] if "center" in config_file.keys() else True
    pad_mode = config_file["pad_mode"] if "pad_mode" in config_file.keys() else  "reflect"

    norm = config_file["norm"] if "norm" in config_file.keys() else "orto"
    dct_type = config_file["dct_type"] if "dct_type" in config_file.keys() else 2

    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode)
    stft = p.abs(stft)**2

    filters = get_filters(fs, n_fft, n_mfcc)

    filtered_stft = np.dot(stft.T, filters.T)

    MFCC = scipy.fftpack.dct(filtered_stft, axis=-2, type=dct_type, norm=norm)[..., :n_mfcc, :]

    return MFCC
