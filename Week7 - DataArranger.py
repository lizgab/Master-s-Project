# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:20:38 2018

@author: mtech
"""

"""
This programme (should) takes the raw data, sorts it into baselines, averages for repeated data, 
undoes the mod 2pi, calculates an array where data needs predicting and prints to a file 
for use in a george library process
"""

import numpy as np
import cmath
import matplotlib.pyplot as plt

#import george
#from george import kernels

# import scipy.optimize as op



#function to read in data 
def readfile(filename):
    #open file
    filedata = open(filename, 'r')
    #create blank arrays 
    timestamp, antenna1, antenna2, vis, visuncert, num, flag, phs = ([] for i in range(8))
    #while not at the end...
    while True:
        line = filedata.readline() #read the lines
        if not line: break #end infinite loop if no more lines
                                   
        items = line.split(',') #split the items in each line by ','
        
        #only take unflagged data (tagged as false)
        #also only takes calibration data (when num=2)
        if any("False" in x for x in items) and (items[5]=='2'):
            timestamp.append(float(items[0]))
            antenna1.append(float(items[1]))
            antenna2.append(float(items[2]))
            vis.append(complex(items[3])) #vis is complex
            visuncert.append(float(items[4]))
            num.append(float(items[5]))
            flag.append(str(items[6])) #flag is a string   
        
                  
    #turn into normal numpy arrays
    timestamp=np.array(timestamp)
    antenna1=np.array(antenna1)
    antenna2=np.array(antenna2)
    vis=np.array(vis)
    visuncert=np.array(visuncert)
    num=np.array(num)
    

    #calculate phase
    for i in range (0,len(vis)):
        phs.append(cmath.phase(vis[i])) #calculate phase
    #turn from list in nnumpy array 
    phs=np.array(phs)

    
    return timestamp, antenna1, antenna2, vis, visuncert, num, flag, phs

#********************************************************************************************
#Main


#PARAMETERS TO BE CHANGED:

    
#Call function to read data
timestamp, antenna1, antenna2, vis, visuncert, num, flag, phs = readfile('A75_data.dat')


#Identify baselines and sort data based on baselines
#Give each baseline a unique number, based on a mathematical mapping
baselines = ((antenna1 + antenna2)*(antenna1 + antenna2 +1))/2 + antenna2


#Zip the baselines, phases and times together for sorting
allThree = np.column_stack((baselines,timestamp, phs))
#sort by the baselines in the first column
allThreeSorted = sorted(allThree,key=lambda x: x[0])
#turn into allThreesorted into numpy array
allThreeSorted = np.array(allThreeSorted)


#split arrays by baseline identifier
baselineArrays=np.split(allThreeSorted, np.where(np.diff(allThreeSorted[:,0]))[0]+1) #split the arrays by when theres a difference in the baseline number



#**********************************************************************************************************
#now get data into correct form and export to a data file
#get an array of the number of baselines
for i in range (0, len(baselineArrays)):
    #Select a single baseline worth of data for each i in loop
    oneBaselineArray = baselineArrays[i]
    t=[] #create blank arrays for time and phase
    p=[]
    for n in range (0,len(oneBaselineArray)):
        t.append(oneBaselineArray[n,1]) #fill up time array from 2nd column of this baselines data array
        p.append(oneBaselineArray[n,2]) #fill up phase array from 3rd column of this baselines data array
    

    #Convert to numpy arrays
    t = np.array(t)
    p = np.array(p)

    #Average over values for repeated times
    # Now average over repeated times...
    # Averaging might have a problem if the two values are ~-pi and pi
    #Identify values, location and occurences of each particular timestamp
    times, indicies, inverse, occurences = np.unique(t, return_index=True , return_inverse=True, return_counts=True)    
    phaseAv = []
    #Loop over every unique time
    for j in range(0,len(times)):
       temp = 0
       #If the time is repeated, average over all corresponding values
       # if occurences[i] != 1:
       #Find where the inverse value is repeated and average over these elements of the array
       index = np.asarray(np.where(inverse == inverse[j])) #find values, convert to array
       #Loop from zero to the number of ocurences
       for k in range(0,len(index.T)):
           temp += p[index[0][k]]
       phaseAv.append(temp/len(index.T))#Phase should be averaged over
           
#   #uncomment to plot all figures
#    plt.figure()
#    plt.scatter(times, phaseAv)
#    plt.show()            

    #Output the data in a form to be used by the GP programme
    #Stitch together data arrays and transpose to columns
    data = np.array([times, phaseAv])
    data = data.T

    #Open a .txt file to write to 
    #format.(i) makes a different file for each baseline thats being looped over
    with open("{}datafile.txt".format(i), 'wb+') as datafile_id:
    #Write the data, formatted and separated by a comma
        np.savetxt(datafile_id, data, fmt=['%.2f','%.2f'], delimiter=',')






















#"""
#PARAMETERS TO CHANGE 
#standardWrapCheck = 2/3
#timeGap = 200
#fillTimeGap = 1.7
#gapWrapCheck = cmath.pi/2
#sampleRate = 0.86
#    #Now undo mod 2pi and work out where to put new datapoint estimates (both in one loop for efficiency)
#    #TO DO - Work out an average gradient of the graph?
#    #For now, hard code some stuff in
#    gap =[]
#    #If consecutive datapoints are more than ~2pi apart then add 2pi to all consecutive datapoints
#    for i in range(0, len(tAv) - 1): #Loop over entire dataset
#        #Check for discontinuity - if phs gap is big then discont
#        #If big time gap with little increase then time gap
#        if pAv[i+1] - pAv[i] > standardWrapCheck*cmath.pi or tAv[i+1] - tAv[i] > timeGap and pAv[i+1] - pAv[i] < gapWrapCheck : 
#            #If discont, add on pi for every subsequent value - every time for every discont
#            for j in range(i + 1, len(pAv)):
#                pAv[j] = pAv[j] + 2*cmath.pi
#        #Check for gaps in data to be filled in by GP
#        if tAv[i+1] - tAv[i] > fillTimeGap: #Check for timegaps larger than two dp
#            #Large timegap, fill in the gap with data appearing at roughtly the sampling rate
#            temp = np.linspace(tAv[i], tAv[i+1], int((tAv[i+1] - tAv[i])/sampleRate))
#            gap = np.append(gap, temp)  #min, max, number of points
#            
#    #Uncomment to plot data after mod 2pi has been sorted (will plot all graphs!)
#   
#    #Also write where the gaps are t a different 
#    with open("{}gapdatafile.txt".format(i), 'wb+') as datafile_id:
#    #Write the data, formatted and separated by a comma
#        np.savetxt(datafile_id, gap, fmt=['%.2f'], delimiter=',')

#    
#"""
#
#    

#
























#calculating parameters is now in next code 
"""
#Now fit this data using a Guassian model
#CALCULATE COVARIANCE MATRIX - Use relevent kernals For now, use two exp kernels
# Squared exponential kernel, takes into account long term rise
k1 = 1**2 * kernels.ExpSquaredKernel(metric=1**2)
#rSquared exponential kernel,
#k2 = 1**2 * kernels.ExpSquaredKernel(metric=100**2)
#combine kernals 
kernel = k1 # + k2
#initiates combines kernels in george library 
gp = george.GP(kernel, mean=np.mean(pAv), fit_mean=True)
#compues covarience matrix - wants an array, not a list
#gp.compute(tAv)
"""


"""
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
"""

"""
#plot results
#plt.figure()
x = np.linspace(min(tAv), max(tAv), 20000)
mu, var = gp.predict(pAv, x, return_var=True)
#std = np.sqrt(var)
#plt.scatter(tAv, pAv)
#plt.fill_between(x, mu+std, mu-std, color="g", alpha=0.5)
"""
