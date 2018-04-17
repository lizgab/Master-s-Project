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
    timestamp, antenna1, antenna2, vis, visuncert, num, flag = ([] for i in range(7))
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

    
    return timestamp, antenna1, antenna2, vis, visuncert, num, flag

#********************************************************************************************
#Main
    
#PARAMETERS TO BE CHANGED:
#CAN CHANGE the i in range 0, len(baselineArrays) on line ~100 to change which baselines to plot data for
phaseDiff = cmath.pi #value that the difference between two phases has to be equal to for the programme to have considered data to have wrapped-around on itself
    
#Call function to read data
timestamp, antenna1, antenna2, vis, visuncert, num, flag = readfile('A75_data.dat')

#Identify baselines and sort data based on baselines
#Give each baseline a unique number, based on a mathematical mapping
baselines = ((antenna1 + antenna2)*(antenna1 + antenna2 +1))/2 + antenna2

#Zip the baselines, phases and times together for sorting
allThree = np.column_stack((baselines,timestamp, vis))
#sort by the baselines in the first column
allThreeSorted = sorted(allThree,key=lambda x: x[0])
#turn into allThreesorted into numpy array
allThreeSorted = np.array(allThreeSorted)

#split arrays by baseline identifier
baselineArrays=np.split(allThreeSorted, np.where(np.diff(allThreeSorted[:,0]))[0]+1) #split the arrays by when theres a difference in the baseline number

#**********************************************************************************************************
#now get data into correct form and export to a data file
#loop over all possible baselines and select relevent data, can also print to file and plot
for i in range (0, len(baselineArrays)): #set max to len(baselineArrays) to go to end
    #Select a single baseline worth of data for each i in loop
    oneBaselineArray = baselineArrays[i]
    t=[] #create blank arrays for time and visibility
    v=[]
    for n in range (0,len(oneBaselineArray)):
        t.append(np.real(oneBaselineArray[n,1])) #fill up time array from 2nd column of this baselines data array
        v.append(oneBaselineArray[n,2]) #fill up phase array from 3rd column of this baselines data array
    
    #Convert to numpy arrays
    t = np.array(t)
    v = np.array(v)
    
    #Calculate phase from visibility
    phs = []
    for j in range (0,len(v)):
            phs.append((cmath.phase(v[j]))) #calculate phase
    #turn from list into numpy array 
    phs=np.array(phs)

    #Plot these if you want - not that these contain two visibility values for some times!
    plt.figure()
    plt.scatter(t, phs)
    plt.show()
    
    #Average over repeated visibility values - same baseline two peices of data at same time
    #NOTE: AVERAGE AS VISIBILITY TO AVIOID WRAPPING PROBLEM CAUSED BY PHASE BEING CIRCULAR
    #Identify values, location and occurences of each particular timestamp
    times, indicies, inverse, occurences = np.unique(t, return_index=True , return_inverse=True, return_counts=True)    
    visAv = []
    #Loop over every unique time - If the time is repeated, average over all corresponding values
    for k in range(0,len(times)):
        temp = 0
        #Find where the inverse value is repeated and average over these elements of the array
        index = np.asarray(np.where(inverse == inverse[k]))
        #Loop from zero to the number of ocurences
        for j in range(0,len(index.T)):
               temp += v[index[0][j]]
        #Add  averaged phase values to array
        av = temp/len(index.T)
        #Add averages to an array
        visAv.append(av)
        
    #Calculate phase from visibility average
    phsAv = []
    for l in range (0,len(visAv)):
            phsAv.append((cmath.phase(visAv[l]))) #calculate phase
    #turn from list into numpy array 
    phsAv=np.array(phsAv)

    #Plot these if you want - now properly averaged!!!
    plt.figure()
    plt.scatter(times, phsAv)
    plt.show()
    


    #Output the data in a form to be used by the GP programme
    #Stitch together data arrays and transpose to columns
    data = np.array([times, phsAv])
    data = data.T
    
    #Open a .txt file to write to 
    #format.(i) makes a different file for each baseline thats being looped over
    with open("{}datafile.txt".format(i), 'wb+') as datafile_id:
    #Write the data, formatted and separated by a comma
        np.savetxt(datafile_id, data, fmt=['%.2f','%.2f'], delimiter=',')







#"""
#Mod 2pi bit!
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
