import soundfile as sf
import numpy as np
from createLPF import createLPF

filterOrder = 120
window = "hamming"
targetFrequency = 16000

Fcut = targetFrequency/2

data, Fs = sf.read('../../Maciej Chrzanowski/1/start.wav')
# print(data.shape)
# print(data.ndim)

# check if wav file is stereo. If so, take 1st channel
if data.ndim == 2:
    data = data[:,0]

# print(data.shape)
# print(data.ndim)
# data = np.append(data,0)
# print(data.shape)

############## downsampling ##############
##filtering
filter = createLPF(Fs, Fcut, filterOrder, window)
filteredData = np.convolve(data, filter)

##decimation 44100*4/11 ~= 16036 --//-- 48000/3 = 16000
if Fs == 44100: #works only if targetFrequency is 16000
    step = 11
    bufferData = [0]
    for i in range(len(filteredData[:-1])): #without last sample
        bufferData.append(filteredData[i])
        ###linear approximation
        bufferData.append((3*filteredData[i] + 1*filteredData[i+1])/4)
        bufferData.append((  filteredData[i] +   filteredData[i+1])/2)
        bufferData.append((1*filteredData[i] + 3*filteredData[i+1])/4)
        ###
    print('bufferData: = ', len(bufferData))
    resampledData = [0]
    for i in range(0,len(bufferData),step):
        resampledData.append(bufferData[i])
    print('resampledData: = ', len(resampledData))
    resampledData.pop(0)
    print('resampledData: = ', len(resampledData))

else: #works if Fs is multiple of 48000
    step = Fs/targetFrequency
    resampledData = [0]
    for i in range(0,len(filteredData),step):
        resampledData.append(filteredData[i])
    resampledData.pop(0)
##'resampledData' is filtered and downsampled to 16kHz
##########################################
