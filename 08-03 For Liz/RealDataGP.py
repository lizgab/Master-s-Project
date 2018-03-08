
import numpy as np
import cmath
import matplotlib.pyplot as plt
import scipy.optimize as op

from xlrd import open_workbook

import george
from george import kernels

#Get data... open up in test spreadsheet
wb = open_workbook('adjustedGap.xlsx')
for Sheet1 in wb.sheets():
    number_of_rows = Sheet1.nrows
    number_of_columns = Sheet1.ncols
    
    pAv = []
    tAv = []
    for row in range(number_of_rows):
        tAv.append(Sheet1.cell(row,0).value)
        pAv.append(Sheet1.cell(row,1).value)

#Convert to arrays
pAv = np.array(pAv)
tAv = np.array(tAv)


#Now fit this data using a Guassian model
#CALCULATE COVARIANCE MATRIX - Use relevent kernals For now, use two exp kernels

# Squared exponential kernel, takes into account long term rise
#k1 = 0.01**2 * kernels.ExpSquaredKernel(metric=10**2)
#rSquared exponential kernel,
k1 = 7**2 * kernels.ExpSquaredKernel(metric=100**2)

#removed kernel 2 as new data set doesnt show periodicity
#k2 = 2.4**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(gamma=2/1.3**2, log_period=0.0)
## rational quadratic kernel for medium term irregularities.
## h = 0.66; alpha = 0.78; beta = 1.2
#k3 = 0.66**2 * kernels.RationalQuadraticKernel(log_alpha=np.log(0.78), metric=1.2**2)
#
## noise kernel: includes correlated noise & uncorrelated noise
## h = 0.18; lambda = 1.6; sigma = 0.19
#k4 = 0.18**2 * kernels.ExpSquaredKernel(1.6**2)

#combine kernals 
kernel = k1 #+ k2  + k3 + k4

#initiates combines kernels in george library 
gp = george.GP(kernel, mean=np.mean(pAv), fit_mean=True)

#compues covarience matrix - wants an array, not a list
gp.compute(tAv)
print(gp.log_likelihood(pAv))

"""
#******************************************************************
#NOW OPTIMISING PARAMETERS 
#theres an optimising function in the scipy library
#need to feed in a function to optimise and the gradient of the function
#the function to optimise is the log liklihood, which is on the george library 
# Define the objective function (negative log-likelihood in this case).
def nll(p):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(pAv, quiet=True)
    return -ll if np.isfinite(ll) else 1e25
# And the gradient of the objective function.
def grad_nll(p):
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(pAv, quiet=True)
# You need to compute the GP once before starting the optimization.
gp.compute(tAv)
# Print the initial ln-likelihood.
print(gp.log_likelihood(pAv))
# Run the optimization routine.
#initial parameter estimates
p0 = gp.get_parameter_vector()
#feed into optimisation routine 
results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
# Update the kernel and print the final log-likelihood.
gp.set_parameter_vector(results.tAv)
print(gp.log_likelihood(pAv))
print(p0)
"""

#plot results
#plt.figure()

#Produce an array which contains time stamps where there should be a datapoint
#TO DO - CURRENTLY HARD-CODED, NEED TO CHANGE THAT!!
gap = np.linspace(tAv[2089], tAv[2090], int((tAv[2090] - tAv[2089])/0.86))

#Predict the values of mean and variance at the locations where there should be data
mu, var = gp.predict(pAv, gap, return_var=True)
std = np.sqrt(var)

#Plot original data and data and mean/var of predicted data
plt.figure()
plt.scatter(gap, mu , marker = ".", label = "Predicted data")
plt.scatter(tAv, pAv, marker = "+", label = "Original data")
plt.fill_between(gap, mu+std, mu-std, color="g", alpha=0.5)
plt.legend()
