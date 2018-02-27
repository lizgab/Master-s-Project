# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:31:23 2018

@author: Piers
"""
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

#function = np.random.uniform(low=1.0, high=1.0, size=100)
#initialize an array of all zeros except "Infinity" (TO DO - COULD MAKE IT INFINITY?) at one point
delta = np.full(100, 0)
delta[50] = 100

#Show this "Delta function"
plt.plot(delta, '.')
plt.show()

#FT this "Delta function"
deltaFT = np.fft.rfft(delta) #TO DO - CHECK NORMALIZATION?? Works out upon IFT but worth knowing

#split this FT into real and imag parts
R = np.real(deltaFT)
I = np.imag(deltaFT)

#Get Amp and phase info from FT data
phs = []
amp = (deltaFT.real**2 + deltaFT.imag**2)**(1/2)
for i in range (0,len(deltaFT)):
    phs.append(cmath.phase(deltaFT[i])) #calculate phase
    

#phs = np.array(phs) #TO DO THIS STEP CONVERTS FROM LIST TO ARRAY AND LOOSES PRECISION

#FT the axis to find the frequency centres
#freq = np.fft.rfftfreq(100)        
        
#plt.plot(freq, amp, 'x')
#plt.plot(freq, phs, '.')
#plt.show()

#Now add some systematic phase offset to the data

phaseMultiplier = 0 #CHANGE ME
newPhase = phs + (math.pi)*(phaseMultiplier) #Add a multiple of pi

#Recombine into real and imaginary parts
newR=[]
newI=[] 
newDeltaFT=[]

#RECONSTRUCT THE FT DATA - back into real and imag parts.
for j in range (0,len(phs)):

    newR = np.append(newR, amp[j]*math.cos(newPhase[j]))
    newI = np.append(newI, amp[j]*math.sin(newPhase[j]))
    
    #newR = np.append(newR, (amp[j]))
    #newI = np.append(newI, amp[j]*newPhase[j])
    
for k in range (0,len(phs)):
    newDeltaFT = np.append(newDeltaFT, complex(newR[k] , newI[k]))

#IFT the result and get "Delta function" back
inverse = np.fft.irfft(newDeltaFT)
plt.plot(inverse, '.')
plt.show()


#All this stuff can give you an actual delta function but it's not too happy with fft it
"""

import numpy as np
import matplotlib.pyplot as plt

#Import stuff used for delta function!
import sympy
from sympy mpy.abc import x

# Test - does the integral act as expected?
print(sympy.integrate(DiracDelta(x), (x, 0, 5.0))) #(it sortof does...)

#FT The delta function
np.fft.fft(DiracDelta(x))

import DiracDelta
from sy
"""

