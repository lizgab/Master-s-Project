# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 09:37:51 2018

@author: mtech
"""

import numpy as np
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        
        #only take unflagged  
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
timestamp, antenna1, antenna2, vis, visuncert, num, flag, phs = readfile('A75_data.dat')

#Identify baselines and sort data based on baselines
baselines = ((antenna1 + antenna2)*(antenna1 + antenna2 +1))/2 + antenna2

#Sort all data into separate baselines and convert into arrays
sortedBaselines = np.array(sorted(baselines))
sortedTimestamp = np.array([x for _,x in sorted(zip(baselines,timestamp))]) 
sortedPhs = np.array([x for _,x in sorted(zip(baselines,phs))])
sortedVis = np.array([x for _,x in sorted(zip(baselines,vis))])

#merge arrays into one to avoid sorting issues 
mergesort= np.zeros((4,len(sortedBaselines))) #blank array
mergesort= np.vstack((sortedBaselines,sortedTimestamp,sortedPhs,sortedVis)) #stack them on top of each other
mergesort= mergesort.T #transpose

#test, test1 = np.unique(sortedBaselines, return_index=True)
arrays=np.split(mergesort, np.where(np.diff(mergesort[:,0]))[0]+1) #split the arrays by when theres a difference in the baseline number

#extract needed info from array of arrays 
t=[]
p=[]
v=[]
a1 = arrays[1]
for n in range (0,len(a1)):
    t.append(a1[n,1])
    p.append(a1[n,2])
    v.append(a1[n,3])
    
np.array(t)
np.array(p)
np.array(v)
    
plt.scatter(t,p)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(np.real(v), np.imag(v), t)
