# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:17:36 2018

@author: Piers
"""

import numpy as np
import matplotlib.pyplot as plt
import george
from george import kernels
import statsmodels.api as sm #Contains the dataset on co2
import scipy.optimize as op

#import the dataset we are using
data = sm.datasets.get_rdataset("co2").data

time = data.loc[:,'time']
co2 = data.loc[:,'co2']

time = time.values

plt.plot(time,co2,marker = ".")

#Model this data in order to predict future behaviour

#Use relevent kernals

# 1) Squared exponential kernel, takes into account long term rise
# Squared exponential kernel
# θ1 = 66; θ2 = 67
theta1 = 66
theta2 = 67
#k1 = pow(theta1,2) * kernels.ExpSquaredKernel(pow(theta2,2))

k1 = 66.0**2 * kernels.ExpSquaredKernel(67.0**2)
gp = george.GP(k1, mean = 0.0)
#compute covar matrix
gp.compute(time)



# =============================================================================
# # range of times for prediction:
# x = np.linspace(max(time), 2025, 2000)
#  
# # calculate expectation and variance at each point:
# mu, cov = gp.predict(time, x)
# std = np.sqrt(np.diag(cov))
# 
# ax = plt.subplot(111)
#  
# # plot the original values
# plt.plot(time,co2,ls=':') 
#  
# # shade in the area inside a one standard deviation bound:
# ax.fill_between(x,mu-std,mu+std,facecolor='lightgrey', lw=0, interpolate=True)
#  
# plt.ylabel(r"CO$_2$ concentration, ppm")
# plt.xlabel("year")
# plt.title(r"Mauna Loa CO$_2$ data - Initial Prediction")
# plt.axis([1958.,2025.,310.,420.])
# plt.grid()
#  
# # display the figure:
# plt.show()
# =============================================================================

