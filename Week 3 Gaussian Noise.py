# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:57:55 2018

@author: Piers
"""

import numpy as np
import matplotlib.pyplot as plt

noise = np.random.normal(0,1,200)

plt.plot(noise)
plt.show()

noiseFT = np.fft.rfft(noise)

noise1 = np.fft.irfft(noiseFT)

plt.plot(noise1)
plt.show()


