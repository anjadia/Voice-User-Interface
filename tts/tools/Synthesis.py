import numpy as np
from numpy.random import randn 
from scipy.signal import lfilter
#import numpy.linalg as lin
from ruamel.yaml import YAML

def ConcatenateFrames(frames, overlap):
    frameCount = np.shape(frames, 1)
    frameLen = np.shape(frames, 2)
    step = int(frameLen*(1-overlap))

    newLen = frameLen + (frameCount-1)*(frameLen-step)
    X = np.zeros(newLen)
    idx = 0
    for i in range(frameCount):
        X[i,idx:frameLen] += frames[i,:]
        idx += step

    return X

def FrameSynthesis(a, g, frameLen, isVoiced):
    noise = randn(frameLen)
    noise = noise/max(abs(noise))
    delta = np.zeros(frameLen)
    delta[0] = 1
    #print(type(g),type(a),type(noise))
    a = np.array(a)
    a = np.concatenate([1], a)
    #print(type(g),type(a),type(noise))
    if isVoiced:
        SynthFrame = lfilter(g,a,delta)
    else:
        SynthFrame = lfilter(g,a,noise)

    return SynthFrame

def Synthesis(a, g, voicedFrames, configFile):
    overlap = configFile["frameOverlap"] if "frameOverlap" in configFile.keys() else 0.5

    frameCount = np.shape(voicedFrames, 1)
    frameLen = np.shape(voicedFrames, 2)
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