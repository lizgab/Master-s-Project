# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:47:21 2018

@author: mtech
"""

import numpy as np
import cmath

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

#Main
timestamp, antenna1, antenna2, vis, visuncert, num, flag, phs = readfile('A75_data.dat')

#Identify baselines and sort data based on baselines
baselines = ((antenna1 + antenna2)*(antenna1 + antenna2 +1))/2 + antenna2

#Sort all data into separate baselines and convert into arrays
sortedBaselines = np.array(sorted(baselines))
sortedTimestamp = np.array([x for _,x in sorted(zip(baselines,timestamp))]) # TO DO - NOT SORTED RIGHT
sortedPhs = np.array([x for _,x in sorted(zip(baselines,phs))]) # TO DO - NOT SORTED RIGHT

b1=[]
t1=[]
p1=[]


#Get each baseline worth of data in separate array
test, test1 = np.unique(sortedBaselines, return_index=True)

for i in range (0, len(test1) - 1):
    for j in range (test1[i], test1[i+1]):
        b1.append(sortedBaselines[i])
        t1.append(sortedTimestamp[i])
        p1.append(sortedPhs[i])
