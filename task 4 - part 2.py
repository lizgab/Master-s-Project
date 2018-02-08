# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:42:31 2018

@author: mtech
"""

import numpy as np
import matplotlib.pyplot as plt
import math

#generates random noise signal
noise = np.random.normal(0,1,200)


#fourier transforms
noiseFT = np.fft.rfft(noise)

#visulise the amplitude
#seperate real and imaginary part of the fft 
Z = (noiseFT.real, noiseFT.imag)
A = noiseFT.real
B = noiseFT.imag

#calculate amplitude and phase 
amp=[]
phase=[]
for i in range (0,len(noiseFT)):
    amp = np.append(amp, ((A[i]**2) + (B[i]**2))**(0.5))
    phase = np.append(phase, math.atan(B[i]/A[i]))

#visualise amplitude n phase 

plt.plot(amp)

plt.title(r"Amplitude")
plt.show()

plt.plot(phase)
plt.title(r"Phase")
plt.show()


#add noise to phase bit only 
#generate random noise signal of smaller amplitude than original noise signal
smallnoise = np.random.normal(0,0.1,200)
noisephase = []
for i in range (0,len(phase)):
    noisephase= np.append(noisephase, (phase[i]+smallnoise[i]))

plt.plot(noisephase)
plt.title(r"Phase with noise")
plt.show()

#get array of real and imaginary from shifted phase 

A_new=[]
B_new=[] 
FT_new=[]

for j in range (0,len(noisephase)):
    A_new = np.append(A_new, (amp[j]**2)/(1+((math.tan(noisephase[j]))**2)))
for i in range (0,len(noisephase)):
    B_new = np.append(B_new, A_new[i]*(math.tan(noisephase[i])))
for k in range (0,len(noisephase)):
    FT_new = np.append(FT_new, complex(A_new[k] , B_new[k]))


