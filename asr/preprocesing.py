import numpy as np
import math
from pathlib import Path
from scipy.io import wavfile
from ruamel.yaml import YAML

###TODO powrzucac te wszystkie funkcje do klasy Preprocessor wg szablonu z asr/tools/preproseccing.py -anjadia

def createLPF(fs, Fcut, n, winName):
    #function creates a n-th order low-pass FIR filter with winName window
    #fs - sampling frequency, Fcut - cut-off frequency
    #fs, Fcut, n - int
    #winName - string

    #returns filter's impulse response
    LPF = np.zeros(n)
    Fratio = Fcut/fs

    for i in range(n):
        if(i != int(n/2) - 1):
            LPF[i] = np.sin(2*np.pi*Fratio*(i - (int(n/2) - 1))) / (np.pi*(i- (int(n/2) - 1)))
        else:
            LPF[i] = 2*Fratio

    win = eval("np." + winName + "(n)")

    return LPF * win


def resample(signal, originalFrequency, config_file):


    filterOrder = config_file["filterOrder"] if "filterOrder" in config_file.keys() else 120
    window = config_file["window"] if "window" in config_file.keys() else "hamming"
    targetFrequency = config_file["targetFrequency"] if "targetFrequency" in config_file.keys() else 16000

    Fcut = targetFrequency/2

    # check if wav file is stereo. If so, take 1st channel
    if signal.ndim == 2:
        signal = signal[:,0]

    ############## resampling ##############
    ##filtering
    filter = createLPF(originalFrequency, Fcut, filterOrder, window)
    filteredSignal = np.convolve(signal, filter)

    #delta - step (index) with which original samples are taken. Notice that 'delta' is NOT an integer
    delta = originalFrequency/targetFrequency
    #prealocating array for output signal
    resampledSignal = np.zeros((math.ceil(len(filteredSignal)*(1/delta)))+1)

    sIdx = 0.0 #original signal's index
    rsIdx = 0  #resampled signal's index

    while sIdx < len(filteredSignal):
        intPart = int(sIdx)         #take integer part of index
        fracPart = sIdx - intPart   #calculate fractional part of index

        #linear interpolation
        if intPart + 1 < len(filteredSignal):#simple check for out-of-bound
            resampledSignal[rsIdx] = filteredSignal[intPart]*(1 - fracPart) + filteredSignal[intPart + 1]*fracPart
        else:
            resampledSignal[rsIdx] = filteredSignal[-1]
        sIdx += delta   #increasing index by 'delta' every loop
        rsIdx += 1

    return resampledSignal, targetFrequency
    ##'resampledData' is filtered and downsampled to target frequency kHz


def zero_cross(frame, fs, f_min = 50, f_max = 8000, threshold = 0.2):
    """
    Parameters
    ----------
    frame
        audio frame
        
    fs
        sample rate [Hz]
    
    f_min
        lower threshold of the speech searching [Hz]
    
    f_max
        upper threshold of the speech searching [Hz]
        
    threshold
        threshold of noise/silence in range 0 - 1 
        
    Returns
    -------
    if_speech
        if there is a speech in frame return 1, otherwise 0
    """

    frame = frame.copy()
    frame[np.abs(frame)<threshold] = 0.0

    zero_crossing = np.abs(np.diff(np.sign(frame)))
    zero_crossing = np.nonzero(zero_crossing)
    if len(zero_crossing[0]) > 0:
        ZCR =  np.mean(np.diff(zero_crossing))*2
        if_speech = 1 if (fs/ZCR > f_min and fs/ZCR < f_max) else 0
        return if_speech
    return 0


def crop_silent_zero_cross(audio, fs, config_file):
    """
    Parameters
    ----------
    frame
        audio frame
        
    fs
        sample rate [Hz]
    
    f_min
        lower threshold of the speech searching [Hz]
    
    f_max
        upper threshold of the speech searching [Hz]
        
    threshold
        threshold of noise/silence in range 0 - 1 
        
    Returns
    -------
    if_speech
        if there is a speech in frame return 1, otherwise 0
    """
    
    amplitude = np.max(np.abs(audio))
    audio_normalized = audio.copy()/np.max(np.abs(audio))
    
    frame_len = config_file["frame_len"] if "frame_len" in config_file.keys() else 0.1
    frame_len = int(np.ceil(frame_len * fs)) 
    frame_count = int(len(audio) / frame_len)  

    audio_normalize_framed = [audio_normalized[i*frame_len:(i+1)*frame_len] for i in range(frame_count)] 
    audio_framed = [audio_normalized[i*frame_len:(i+1)*frame_len] for i in range(frame_count)] 
    
    f_min = config_file["f_min"] if "f_min" in config_file.keys() else 50
    f_max = config_file["f_max"] if "f_max" in config_file.keys() else 8000
    threshold = config_file["threshold"] if "threshold" in config_file.keys() else 0.2

    zc = []
    for frame in audio_normalize_framed:
        zc.append(zero_cross(frame, fs, f_min = f_min, f_max = f_max, threshold = threshold))
        
    audio_vad = []
    audio_silence = []
    for frame, vad in zip(audio_framed, zc):
        if vad == 1:
            audio_vad.extend(frame)
        else: 
            audio_silence.extend(frame)
            
    return np.array(audio_vad), np.array(audio_silence)


def VAD(signal, fs, config_file):
    """ On given signal returns only frames with speech """
    clean_audio, _ = crop_silent_zero_cross(signal, fs, config_file)
    return clean_audio


def normalize(signal):
    """ Normalize to rms """

    return signal / np.sqrt(np.mean(np.power(signal,2)))


class Preprocessor():

    def __init__(self, config_file):
        self.config_file = config_file

        if all(item in ["resampling", "VAD"] for item in  config_file.keys()):
            print("******\nLoaded config file.\n******")
        else:                
            print("Incorrect config file")
            raise 


    def audio_peprocessing(self, signal, fs):
        signal, fs = resample(signal, fs, self.config_file["resampling"])
        signal = normalize(signal)
        signal = VAD(signal, fs, self.config_file["VAD"])
        return signal, fs


    # def audio_peprocessing(self, signal_path):
    
    #     fs, signal = wavfile.read(signal_path)

    #     signal, fs = resample(signal, fs, self.config_file["resampling"])
    #     signal = normalize(signal)
    #     signal = VAD(signal, fs, self.config_file["VAD"])
    #     return signal, fs


# if __name__ == "__main__":
#     import sys
#     import time

#     wav = sys.argv[1]

#     config_file = Path(__file__).absolute().parents[1] / "config.yaml"

#     with config_file.open() as f:
#         config = YAML(typ="safe").load(f)

#         preprocessor = Preprocessor(config["preprocessing"])

#         start = time.time()

#         audio, fs = preprocessor.audio_peprocessing(wav)
#         wavfile.write("test.wav", 16000, audio)
#         end = time.time()
#         print("Processed in %1.2f sec" % (end - start))

# using: python preprocessing.py <path>
# example: python preprocesing.py /home/anjadia/VOI/2020/nieznani/9/1/9_1\'naprzód\'.wav
