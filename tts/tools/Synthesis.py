import numpy as np
from numpy.random import randn 
from scipy.signal import lfilter
from preprocesing import createLPF
#import numpy.linalg as lin
#from ruamel.yaml import YAML

def ConcatenateFrames(frames, overlap):
    shape = np.shape(frames)
    frameCount = shape[0]
    frameLen = shape[1]
    step = int(frameLen*(1-overlap))

    newLen = frameLen + (frameCount-1)*(frameLen-step)
    X = np.zeros(newLen)
    idx = 0
    for i in range(frameCount):
        X[idx:idx+frameLen] += frames[i,:]
        idx += step

    return X

def FrameSynthesis(a, g, frameLen, isVoiced):
    filter = createLPF(16000,8000,120,"hamming")
    noise = randn(frameLen)
    noise = np.convolve(noise, filter)
    noise = noise[0:frameLen]
    noise = noise/max(abs(noise))
    delta = np.zeros(frameLen)
    delta[0] = 1
    #print(type(g),type(a),type(noise))
    a = np.array(a)
    a = np.concatenate(([1], a))
    #print(type(g),type(a),type(noise))
    #print(np.shape(g),np.shape(a),np.shape(noise))
    if isVoiced:
        SynthFrame = lfilter([g],a,delta)
    else:
        SynthFrame = lfilter([g],a,noise)

    return SynthFrame

def Synthesis(a, g, voicedFrames, frameLen, configFile):
    overlap = configFile["frameOverlap"] if "frameOverlap" in configFile.keys() else 0.5
    frameCount = len(voicedFrames)
    SynthFrames = np.zeros([frameCount, frameLen])

    for i in range(frameCount):
        SynthFrames[i,:] = FrameSynthesis(a[i], g[i], frameLen, voicedFrames[i])

    SynthesisedVoice = ConcatenateFrames(SynthFrames, overlap)

    return SynthesisedVoice


##############   tests   #################
# noise = FrameSynthesis([0.39421364, 0.17460218, -0.51700188, -0.32923405, -0.32796281, -0.20344663, 0.19133089, 0.06823649, -0.12269852, -0.31750758],[-0.02572850648259036],320, 1)
# print(noise)
# print(np.shape(noise))



#calculated aCoeff --> 1, 0.39421364, 0.17460218, -0.51700188, -0.32923405, -0.32796281, -0.20344663, 0.19133089, 0.06823649, -0.12269852, -0.31750758
#and gain --> -0.02572850648259036