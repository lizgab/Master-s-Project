"""
Created on Tue Feb 27 10:47:21 2018

@author: mtech
"""

import numpy as np
import math

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
       

        if any("False" in s for s in items):
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
    #flag=np.array(flag)
    

    
    return timestamp, antenna1, antenna2, vis, visuncert, num, flag

#Main
timestamp, antenna1, antenna2, vis, visuncert, num, flag = readfile('A75_data.dat')









