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
    rows=len(a)
    columns=len(b)
    matrix= np.zeros((rows,columns))
    #fill in rows and columns of this blank matrix
    #ai, bj 
    #size n x m
    for i in range (rows):
        for j in range (columns):
            matrix[i,j]= kernelfunction(h=1,ai=a[i],bj=b[j],l=l)
    return matrix;






#generate evenly spaced data
#Set range to generage data in
low = 0
high = 100
sampNum = 300
l = 20

#Generate array for covar kernal
x=np.linspace(low,high,sampNum)


#calculate matrix of kernels 
#plot matrix
covmatrix= matrixfunction(x,x,l)
plt.imshow(covmatrix, cmap='hot', interpolation='nearest')
plt.show()
    

#generate covarient gaussian distrubution 
#define needed mean matrix for random multivariate function
meanlen=len(covmatrix)
meanmatrix= np.zeros((meanlen))


#covariate distrubution sampling
#Draw random samples from a multivariate normal distribution
#output gives each column of matrix as gaussian
gaussianmatrix = np.random.multivariate_normal(meanmatrix,covmatrix,5)


#plot samples 
numgauss=5
for k in range (0,numgauss):
    gaussian = gaussianmatrix[k,:]
    plt.plot(x,gaussian)
    
#Select a column as each column is a gaussian
gauss1 = gaussianmatrix[0,:]

#plt.xlim((0,20))
#plt.ylim((-3,3))
plt.show()

### Sample and reconstruct data from samples ###
#Select 5 samples from our gaussians
samples = np.random.randint(low, sampNum, size=5)
xsamp = x[samples]
samp = gauss1[samples]

#initialise arrays
meanest = []
SDest = []

for i in range (low,sampNum):   
    #Calculate all new covmatrix for our samples for use in mean and SD calc
    covmatrixnew = matrixfunction(xsamp,xsamp,l) #Sample covmatrix
    unknownCovMatrix = matrixfunction(xsamp,[x[i]],l) #Sample and unknown point covmatrix
    unknownKernel = matrixfunction([x[i]],[x[i]],l)#Kernel of unknown with itself
    
    #Calculate all relevent inverses and fot products
    #Calculate inverse of inital covar matrix
    covMatrixInv = np.linalg.inv(covmatrixnew)
    #Calculate the mean value for the estimated point i.e K*Trans.KInv.Xval
    firstdot = (np.dot(unknownCovMatrix.T,covMatrixInv))
    
    #calculate mean and SD
    tempMeanest = (np.dot(firstdot,samp.T))
    tempSDest = -(np.dot(firstdot,unknownCovMatrix)) + unknownKernel
    
    #Append to arrays
    meanest = np.append(meanest, tempMeanest)
    SDest = np.append(SDest, tempSDest)

plt.fill_between(x, meanest + SDest, meanest - SDest)
plt.scatter(x, meanest, c='m', marker=".")
#plt.scatter(x, gauss1, c='g', marker="s")
plt.scatter(xsamp, samp, c='r', marker="s")
plt.show()




