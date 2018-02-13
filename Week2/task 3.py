# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
#"""
#Task 3
#forward predicting Co2 data 
#useful links 
#https://allofyourbases.com/2017/09/17/predicting-the-future/
#http://george.readthedocs.io/en/latest/tutorials/hyper/


#import things i need
import numpy as np
import matplotlib.pyplot as pl
import george
import scipy.optimize as op
from george import kernels


# this is a function to read the Mauna Loa data from file
def read_co2(filename):
 
    co2file = open(filename,'r')
 
    time=[];co2=[]
    while True:
        line = co2file.readline()
        if not line: break
 
        items = line.split()
 
        if (items[0]!='#') and (items[2]!='-99.99'):
 
            time.append(float(items[0]))
            co2.append(float(items[1]))
 
    time=np.array(time)
    co2=np.array(co2)
 
    return time,co2
 
    
t,y = read_co2("mauna_loa.txt")

#shortern data set so can compare current and final trends
y_to_2003 = y[np.where(t<2003.)]
t_to_2003 = t[np.where(t<2003.)]


#**************************************************************************
#CALCULATE COVARIANCE MATRIX
#Model this data in order to predict future behaviour
#Use relevent kernals

# 1) Squared exponential kernel, takes into account long term rise
# Squared exponential kernel
# h = 66; lambda = 67
k1 = 66.0**2 * kernels.ExpSquaredKernel(67.0**2)

#removed kernel 2 as new data set doesnt show periodicity

# rational quadratic kernel for medium term irregularities.
# h = 0.66; alpha = 0.78; beta = 1.2
k3 = 0.66**2 * kernels.RationalQuadraticKernel(0.78, 1.2**2)


# noise kernel: includes correlated noise & uncorrelated noise
# h = 0.18; lambda = 1.6; sigma = 0.19
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2)

#combine kernels
kernel = k1  + k3 + k4

#initiates combines kernels in george library 
#add white noise in this function in this version of george
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
               white_noise=np.log(0.19**2), fit_white_noise=True)
#computes covarience matrix
gp.compute(t_to_2003)

#******************************************************************
#NOW OPTIMISING PARAMETERS 
#theres an optimising function in the scipy library
#need to feed in a function to optimise and the gradient of the function
#the function to optimise is the log liklihood, which is on the george library 

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y_to_2003, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y_to_2003, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(t_to_2003)

# Print the initial ln-likelihood.
print(gp.log_likelihood(y_to_2003))

# Run the optimization routine.
#initial parameter estimates
p0 = gp.get_parameter_vector()
#feed into optimisation routine 
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")


# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(gp.log_likelihood(y_to_2003))


#plot out results 
x = np.linspace(max(t_to_2003), 2025, 2010)
mu, var = gp.predict(y_to_2003, x, return_var=True)
std = np.sqrt(var)

pl.plot(t_to_2003, y_to_2003, ".k")
pl.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

pl.xlim(t.min(), 2025)
pl.xlabel("year")
pl.ylabel("CO$_2$ in ppm");





