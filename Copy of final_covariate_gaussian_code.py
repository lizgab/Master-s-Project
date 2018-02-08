# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import things i need
import math
import numpy as np
import matplotlib.pyplot as plt


#matrix= np.zeros((2,5))


##function to calculate gaussian covarience kernel 
def kernelfunction (h, ai, bj, l):
    k = (pow(h,2))*math.exp(-pow(((ai-bj)/l),2))
    #print (k)
    return k;



#function to fill a matrix with corresponding kernel values
def matrixfunction(a,b,l):
    #make matrix size (input array a x input array b)
    m=len(a)
    n=len(b)
    matrix= np.zeros((n,m))
    #fill in rows and columns of this blank matrix
    #ai, bj 
    #size n x m
    for j in range (0,m):
        for i in range (0,n):
            matrix[i,j]= kernelfunction(h=1,ai=a[i],bj=b[j],l=l)
    return matrix;




#generate random data

x=np.linspace(0,20,1000)

#calculate matrix of kernels 

covmatrix= matrixfunction(x,x,5)
plt.imshow(covmatrix, cmap='hot', interpolation='nearest')
plt.show()
    

#generate covarient gaussian distrubution 
#define needed mean matrix for random multivariate function
meanlen=len(covmatrix)
meanmatrix= np.zeros((meanlen))


#covariate distrubution sampling
#Draw random samples from a multivariate normal distribution
#output gives each column of matrix as gaussian

gaussianmatrix = np.random.multivariate_normal(meanmatrix,covmatrix, 5)


#plot samples 
numgauss=5
for k in range (0,numgauss):
    gaussian = gaussianmatrix[k,:]
    plt.plot(x,gaussian)
    
#plt.xlim((0,20))
#plt.ylim((-3,3))
plt.show()





