# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 12:59:01 2018

@author: mtech
"""

import numpy as np
import cmath
import matplotlib.pyplot as plt
import scipy.optimize as op
from xlrd import open_workbook
import george
from george import kernels

#Get data from files 
#tav pav and gap
#function to read in data 
def readfile(filename):
    #open file
    filedata = open(filename, 'r')
    #create blank arrays 
    tAv, pAv = ([] for i in range(2))
    #while not at the end...
    while True:
        line = filedata.readline() #read the lines
        if not line: break #end infinite loop if no more lines
                                   
        items = line.split(',') #split the items in each line by ','
        
        tAv.append(float(items[0]))
        pAv.append(float(items[1]))
        

    return tAv, pAv



#********************************************************************************************
#Main
#PARAMETERS TO CHANGE 
baselineNum = 0
sampleRate = 100

#read in data from selected baseline
tAv, pAv = readfile("{}datafile.txt".format(baselineNum))

#TEST STUFF - REDUCE SIZE OF DATASET FOR USE IN OPTIMIZAION
tAv = tAv[0:200]
pAv = pAv[0:200]

#Now fit this data using a Guassian model
#CALCULATE COVARIANCE MATRIX - Use relevent kernals

# Squared exponential kernel, takes into account long term rise
k1 = 0.1**2 * kernels.RationalQuadraticKernel(log_alpha = (0.0)**2, metric=(23280)**2)

#long term periodicity
k2 = 0.5**2 * kernels.ExpSquaredKernel(3000**2) * kernels.ExpSine2Kernel(gamma=2**2, log_period=0.0)

# rational quadratic kernel for medium term irregularities.
k3 = 0.7**2 * kernels.RationalQuadraticKernel(log_alpha = (0.0)**2, metric=(1)**2)

#short term periodicity
k4 = 0.3**2 * kernels.ExpSquaredKernel(300**2) * kernels.ExpSine2Kernel(gamma=2**2, log_period=0.0)

#combine kernels
kernel = k1 + k2 + k3 + k4

#initiates combines kernels in george library 
gp = george.GP(kernel, mean=np.mean(pAv), fit_mean=True)

#compues covarience matrix - wants an array, not a list
gp.compute(tAv)
print(gp.log_likelihood(pAv))


#NOW OPTIMISING PARAMETERS 
#theres an optimising function in the scipy library
#need to feed in a function to optimise and the gradient of the function
#the function to optimise is the log liklihood, which is on the george library 
# Define the objective function (negative log-likelihood in this case).
def nll(prob):
    gp.set_parameter_vector(prob)
    ll = gp.log_likelihood(pAv, quiet=True)
    return -ll if np.isfinite(ll) else 1e25
# And the gradient of the objective function.
def grad_nll(prob):
    gp.set_parameter_vector(prob)
    return -gp.grad_log_likelihood(pAv, quiet=True)

# Run the optimization routine.
# Initial parameter estimates
p0 = gp.get_parameter_vector()
# Feed into optimisation routine
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
# Update the kernel and print the final log-likelihood. (results.x are the optimized hyperparams)
gp.set_parameter_vector(results.x)
print(gp.log_likelihood(pAv))

#plot out results 
#Produce an array which contains time stamps where there should be a datapoint
x = np.linspace(min(tAv), max(tAv), sampleRate)

#Predict the values of mean and variance at the chosen point
mu, var = gp.predict(pAv, x, return_var=True)
std = np.sqrt(var)

#Plot original data and data and mean/var of predicted data
plt.figure()
plt.scatter(x, mu , marker = ".", label = "Predicted data")
plt.scatter(tAv, pAv, marker = "+", label = "Original data")
plt.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)
plt.legend()


#
#def gapfile(filename):
#    #open file
#    filedata = open(filename, 'r')
#    #create blank arrays 
#    gap = []
#    #while not at the end...
#    while True:
#        line = filedata.readline() #read the lines
#        if not line: break #end infinite loop if no more lines
#                                   
#        items = line.split(',') #split the items in each line by ','        
#        gap.append(float(items[0]))
#
#        
#    #turn into normal numpy arrays
#    gap=np.array(gap)
#    
#    return gap
#gap = gapfile("{}datafile.txt".format(baselineNum))
