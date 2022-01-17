import numpy as np
import numpy.linalg as lin
from ruamel.yaml import YAML

#function returns array ([nr of frames, frame length]) containing input signal devided into frames,
#as well as nr of frames and frame length
def CreateOverlapingFrames(signal, fs, frameLen, overlap):

    frameLenSamp = int(fs*frameLen) #frame's length in samples
    step = int(frameLenSamp*(1-overlap)) #step between frames' starting points (in samples)
    overlapSamp = frameLenSamp - step #overlap in samples

    isZero =  np.mod(len(signal)-overlapSamp,step) == 0 #check if frames exceed signal length

    if ~isZero: #do if signal needs zero padding
        zerosToAdd = step - np.mod(len(signal)-overlapSamp,step) #calculate how many zeros have to be added at the and of 'signal'
        zeroPadding = np.zeros(zerosToAdd)
        signal = np.concatenate((signal,zeroPadding), axis=None)

    frameCount = int((len(signal)-frameLenSamp) / (step)) + 1 #calculation of the number of frames into which the signal will be divided
    frames = np.zeros([frameCount, frameLenSamp])
    idx = 0
    #win = np.hamming(frameLenSamp)
    for i in range(frameCount):
        frames[i,:] = signal[idx:idx+frameLenSamp]
        idx += step
    
    return frames, frameCount, frameLenSamp


def isVoiced(frames, nFrames, frameLen, window, Threshold, fs):

    voicedFrames = np.zeros(nFrames) #1-voiced, 0-unvoiced
    #fr = np.copy(frames)
    r = np.zeros(frameLen)
    fmin = 1000 #? wartość z książki 500 przy fs=8kHz
    K = int(fs/fmin)

    for i in range(nFrames):
        frame = np.copy(frames[i,:]*window)
        P = 0.3*max(frame)
        #frame thresholding
        frame[((frame<P) & (frame>(-P)))] *= 0
        frame[frame>=P] -= P
        frame[frame<=(-P)] += P

        for k in range(frameLen):
            r[k] = sum(frame[0:frameLen-k]*frame[k:frameLen])

        if (max(r[K:])>(Threshold*r[0])):
            voicedFrames[i] = 1 #frame is voiced

    return voicedFrames

#function finds LPC coefficients from given 'frame'
#returns LPC coefficients and gain (needed for filter in synthesis)
def LPCCoefficients(frame, frameLen, window):
    p = 10 #number of coefficients
    fr = frame * window
    temp = np.zeros(p+1)
    R = np.zeros([p,p])

    for i in range(p+1):
        temp[i] = sum(fr[0:frameLen-i]*fr[i:frameLen])
    r = np.copy(temp[1::])

    for i in range(p):
        for j in range(p):
            R[i,j] = temp[np.abs(i-j)]
    #print('R', len(R))
    a = -1*np.matmul(lin.inv(R),r.T) #solving the matrix equation
    G = r[0] + sum(a[1:p]*r[1:p])

    return a, G

def EncodeLPC(signal, configFile):
    frameLen = configFile['tts']["frameLen"] if "frameLen" in configFile['tts'].keys() else 0.02
    overlap = configFile['tts']["frameOverlap"] if "frameOverlap" in configFile['tts'].keys() else 0.5
    fs = configFile["preprocessing"]["resampling"]["targetFrequency"] if "targetFrequency" in configFile["preprocessing"]["resampling"].keys() else 16000
    Threshold = configFile['tts']["Threshold"] if "Threshold" in configFile['tts'].keys() else 0.35
    p = 10

    frames, frameCount, frameLenSamp = CreateOverlapingFrames(signal, fs, frameLen, overlap)
    print('frameCount', frameCount)
    window = np.hamming(frameLenSamp)
    voicedFrames = isVoiced(frames, frameCount, frameLenSamp, window, Threshold, fs)

    aCoeff = np.zeros([frameCount, p])
    gain = np.zeros(frameCount)
    #print(frames[96:])
    for i in range(frameCount):
        #if i!=98:
        aCoeff[i,:], gain[i] = LPCCoefficients(frames[i,:], frameLenSamp, window)
        #print('i ', i)

    return aCoeff, gain, voicedFrames, frameLenSamp

#########################################################   tests   ###############################
# from scipy.io import wavfile
# fs, signal = wavfile.read('C:/Users/Maciek/Desktop/Student/IA/semestr 2/Interfejs głosowy/kod/out.wav')

# fr,count,frameLen = CreateOverlapingFrames(signal, fs, 0.02, 0.5)
# # # print('fs=',fs)
# # # print(np.shape(fr))
# a,g = LPCCoefficients(fr[0],frameLen,np.hamming(frameLen))
# print('a=',a)
# print('g=',g)

# voicedfr = isVoiced(fr,count,frameLen,np.hamming(frameLen),0.35,fs)
# print(voicedfr)
# print(np.shape(voicedfr))