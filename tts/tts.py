import numpy as np
from scipy.io import wavfile
from ruamel.yaml import YAML
#from pathlib import Path
from preprocesing import Preprocessor
from tools.Encode import EncodeLPC
from tools.Synthesis import Synthesis

#config_file = Path(__file__).absolute().parents[1] / "config.yaml"
#audioPath = Path(__file__).absolute().parents[3] / "2/2/pazdziernik.wav"

#fs, signal = wavfile.read(audioPath)
#print(len(signal))

def processTTS(audioPath, config_file, synthVoicePath):

    preprocessor = Preprocessor()
    fs, signal = wavfile.read(audioPath)

    with config_file.open() as f:
        config = YAML(typ="safe").load(f)
        preprocessedSignal, resampledFs = preprocessor.audio_peprocessing(signal, fs)
        aCoeff, gain, voicedFrames, frameLen = EncodeLPC(preprocessedSignal, config)
        SynthVoice = Synthesis(aCoeff, gain, voicedFrames, frameLen, config)

        zero = np.zeros(2000)
        SynthVoice = np.concatenate((zero,SynthVoice,zero))
        wavfile.write(synthVoicePath,resampledFs,SynthVoice)

