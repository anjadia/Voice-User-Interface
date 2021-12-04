import numpy as np

#function creates a n-th order low-pass FIR filter with winName window
#Fs - sampling frequency, Fcut - cut-off frequency
#Fs, Fcut, n - int
#winName - string

#returns filter's impulse response

def createLPF(Fs, Fcut, n, winName):
    LPF = np.zeros(n)
    Fratio = Fcut/Fs

    for i in range(n):
        if(i != int(n/2) - 1):
            LPF[i] = np.sin(2*np.pi*Fratio*(i - (int(n/2) - 1))) / (np.pi*(i- (int(n/2) - 1)))
        else:
            LPF[i] = 2*Fratio

    win = eval("np." + winName + "(n)")

    return LPF * win
