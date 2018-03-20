"""
This programme (should) takes the raw data, sorts it into baselines, averages for repeated data, 
It will then ... undo the mod 2pi, calculates an array where data needs predicting and prints to a file 
for use in a george library process
"""

import numpy as np
import cmath
import matplotlib.pyplot as plt

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

        #only take unflagged and scan number 2 (This is calib scan)
        if any("False" in x for x in items):
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
#Call function to read data
timestamp, antenna1, antenna2, vis, visuncert, num, flag, phs = readfile('A75_data.dat')

#Assign each baseline a unique number
baselines = ((antenna1 + antenna2)*(antenna1 + antenna2 +1))/2 + antenna2

#Prep arrays
timestampBaseline = []
phaseBaseline = []
numBaseline = []

#Loop over whole dataset looking for baseline[0] data and store it
for i in range (0,len(timestamp)):
  if (baselines[i] == baselines[0] and num[i] == 2): #TO DO - PUT THIS 2 AT THE TOP SOMEWHERE #if the baseline is the same as the first baseline and the scan is number two
      #populate arrays with baseline data
      timestampBaseline.append(timestamp[i])
      phaseBaseline.append(phs[i])  
      numBaseline.append(num[i])

"""
#Won't filter out data to just the calibration data
#Loop over whole dataset looking for baseline[0] data and store it
for i in range (0,len(timestamp)):
  if (baselines[i] == baselines[0]): #if the baseline is the same as the first baseline
      #populate arrays with baseline data
      timestampBaseline.append(timestamp[i])
      phaseBaseline.append(phs[i])  
      numBaseline.append(num[i])
#Check what the column num does - colour datapoints depending on the value of num
"""

#plot, for first baseline, phase against time
plt.figure()
plt.scatter(timestampBaseline,phaseBaseline, c = numBaseline)
plt.show()

#Does the same as the above block of code, except in a slightly different way
#Gives same result
"""
for i in range (0,len(antenna1)):
    if (antenna1[i] == 0 and antenna2[i] == 1):
        timestampReduced.append(timestamp[i])
        phaseReduced.append(phs[i])  
        visReduced.append(vis[i])
        
plt.figure()
plt.scatter(timestampReduced,phaseReduced)
plt.show()
"""


# Now average over repeated times...
# Averaging might have a problem if the two values are ~-pi and pi
times = []
pAv = []
temp = 0
#Identify values, location and occurences of each particular timestamp
times, indicies, occurences = np.unique(timestampBaseline, return_index=True, return_counts=True)

"""

#Identify where values are repeated and average them if they are
# TO DO: THIS IS CURRENTLY NOT WORKING - FIX!
for i in range (0, len(times)):
    for j in range (indicies[0],len(timestampReduced)):
        if (times[0] == timestampReduced[j]):
            temp += phaseReduced[j]
        pAv.append(temp)
        temp = 0

plt.figure()
plt.scatter(times,pAv)
plt.show()
"""
