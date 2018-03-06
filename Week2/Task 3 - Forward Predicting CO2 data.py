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
from statsmodels.datasets import co2

data = co2.load_pandas().data
t = 2000 + (np.array(data.index.to_julian_date()) - 2451545.0) / 365.25
y = np.array(data.co2)
m = np.isfinite(t) & np.isfinite(y) & (t < 1996)
t, y = t[m][::4], y[m][::4]



#**************************************************************************
#CALCULATE COVARIANCE MATRIX
#Model this data in order to predict future behaviour
#Use relevent kernals

# 1) Squared exponential kernel, takes into account long term rise
# Squared exponential kernel
# h = 66; lambda = 67
k1 = 66**2 * kernels.ExpSquaredKernel(metric=67**2)

#removed kernel 2 as new data set doesnt show periodicity
k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(gamma=2/1.3**2, log_period=0.0)
# rational quadratic kernel for medium term irregularities.
# h = 0.66; alpha = 0.78; beta = 1.2
k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)

# noise kernel: includes correlated noise & uncorrelated noise
# h = 0.18; lambda = 1.6; sigma = 0.19
k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2)

#combine kernals 
kernel = k1 + k2  + k3 + k4

#initiates combines kernels in george library 
#add white noise in this function in this version of george
gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
               white_noise=np.log(0.19**2), fit_white_noise=True)
#computes covarience matrix
gp.compute(t)

#******************************************************************
#NOW OPTIMISING PARAMETERS 
#theres an optimising function in the scipy library
#need to feed in a function to optimise and the gradient of the function
#the function to optimise is the log liklihood, which is on the george library 

# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

# You need to compute the GP once before starting the optimization.
gp.compute(t)

# Print the initial ln-likelihood.
print(gp.log_likelihood(y))

# Run the optimization routine.
#initial parameter estimates
p0 = gp.get_parameter_vector()
#feed into optimisation routine 
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")


# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.x)
print(gp.log_likelihood(y))


#plot out results 
x = np.linspace(max(t), 2025, 2010)
#Use covar matrix to predict means and variances at points with gaps in data
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)

pl.plot(t, y, ".k")
pl.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)

pl.xlim(t.min(), 2025)
pl.xlabel("year")
pl.ylabel("CO$_2$ in ppm");




