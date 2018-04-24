"""
This programme (should) takes the raw data, sorts it into baselines, averages for repeated data, 
undoes the mod 2pi, calculates an array where data needs predicting and prints to a file 
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
            visuncert.append(complex(items[4]))
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
GapSpacing = 100 #threshold spacing of the gaps, 
avNum = 2 #number of data points to average
    
#Call function to read data
timestamp, antenna1, antenna2, vis, visuncert, num, flag = readfile('A75_data.dat')
# Update imaginary part of visuncert
visuncert.imag = visuncert.real

#Identify baselines and sort data based on baselines
#Give each baseline a unique number, based on a mathematical mapping
baselines = ((antenna1 + antenna2)*(antenna1 + antenna2 +1))/2 + antenna2

#Zip the baselines, phases and times together for sorting
allfour = np.column_stack((baselines, timestamp, vis, visuncert))
#sort by the baselines in the first column
allFourSorted = sorted(allfour,key=lambda x: x[0])
#turn into allThreesorted into numpy array
allFourSorted = np.array(allFourSorted)

#split arrays by baseline identifier
baselineArrays=np.split(allFourSorted, np.where(np.diff(allFourSorted[:,0]))[0]+1) #split the arrays by when theres a difference in the baseline number

#**********************************************************************************************************
#now get data into correct form and export to a data file
#loop over all possible baselines and select relevent data, can also print to file and plot
for i in range (0, 2): #set max to len(baselineArrays) to go to end
    #Select a single baseline worth of data for each i in loop
    oneBaselineArray = baselineArrays[i]
    time=[] #create blank arrays for time, visibility and error
    vis=[]
    visUncert=[]
    for n in range (0,len(oneBaselineArray)):
        time.append(np.real(oneBaselineArray[n,1])) #fill up time array from 2nd column of this baselines data array
        vis.append(oneBaselineArray[n,2]) #fill up phase array from 3rd column of this baselines data array
        visUncert.append(oneBaselineArray[n,3]) # fill up uncertainty array from 4th column
    
    #Convert to numpy arrays
    time = np.array(time)
    vis = np.array(vis)
    visUncert = np.array(visUncert)
    
    # Calculates phase BEFORE average and prints, useful for demo (no errors tho)
    """
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
    """
    
    #Average over repeated visibility values - same baseline two peices of data at same time
    #NOTE: AVERAGE AS VISIBILITY TO AVIOID WRAPPING PROBLEM CAUSED BY PHASE BEING CIRCULAR
    #Identify values, location and occurences of each particular timestamp
    times, indicies, inverse, occurences = np.unique(time, return_index=True , return_inverse=True, return_counts=True)    
    visAv = []
    visUncertAv = []
    #Loop over every unique time - If the time is repeated, average over all corresponding values
    for k in range(0,len(times)):
        temp = 0
        temp1 = 0
        #Find where the inverse value is repeated and average over these elements of the array
        index = np.asarray(np.where(inverse == inverse[k]))
        #Loop from zero to the number of ocurences adding the values ready for average
        for j in range(0,len(index.T)):
               temp += vis[index[0][j]]
               temp1 += visUncert[index[0][j]]
        #Add  averaged phase and error values to array
        av = temp/len(index.T)
        avUncert = temp1/len(index.T)
        #Add averages to an array
        visAv.append(av)
        visUncertAv.append(avUncert)
        
    #GAP DESIGNATION
    #this section isolates each chunk of data and tells you the final point before the gap in the array labeled blockEnd
    blockEnd = [] 
    #for entire data set
    for h in range(0, len(times)-1):
        #if the spacing between two consequtive points is greater than he threshold set in the variable GapSpacing
        if( (times[h+1]-times[h]) > GapSpacing):
            #add the location to array
            blockEnd.append(h)
    blockEnd = np.array(blockEnd) #convert to numpy array
    blockEnd = np.insert(blockEnd, 0, 0) #add an initial value of zero (useful in next bit of program)
    
    
    #AVERAGING EVERY N VALUES
    #this section ensures if integer number of N dont fit into a certain gap, 
    #the final values from that gap that dont fit are discarded
    #this will ensure a consistent frequency sampling rate and that averages are not taken over gaps
    #sometimes 4736/4742....depending on deleting through loop or after.. look into this
    deleteTimesArr=[]
    for k in range (0, len(blockEnd)-1): #over all the gaps
        if ((blockEnd[k+1]-blockEnd[k])/avNum).is_integer(): #if an integer number of avNums fit into the gap, do nothing
            print('integer')
        else: 
            #if not an integer number of averages in gap.. 
            remainder=(blockEnd[k+1]-blockEnd[k])%avNum #calculates the remaining number of values after integer 
            deleteTimes= list(range(((blockEnd[k+1])-remainder),blockEnd[k+1]+1)) #creates list of values to delete
            deleteTimes= np.array(deleteTimes) #turns list to array
            deleteTimesArr.append(deleteTimes) #adds to big array of arrays of all values to be deleted
            

    deleteTimesArr=np.concatenate( deleteTimesArr, axis=0 )  #turns array of arrays into single array   
    times = np.delete(times,deleteTimesArr) #updates times
    visAv = np.delete(visAv,deleteTimesArr) #updates visability 
    visUncertAv = np.delete(visUncertAv, deleteTimesArr)       
    
    #average multiple data points as specified by avnum variable
    #adapted from https://stackoverflow.com/questions/40217015/averaging-over-every-n-elements-of-an-array-without-numpy
    visAvReduced = []
    timesReduced = [] 
    visUncertReduced = []
    for k in range(len(visAv)//avNum): #sets number of averages
        new_value_v = 0 
        new_value_t = 0
        new_value_u = 0
        for j in range(avNum):
            new_value_v += visAv[k*avNum + j] #fills array with avNum variables at correct point 
            new_value_t += times[k*avNum + j] 
            new_value_u += visUncertAv[k*avNum + j] 
            
        visAvReduced.append(new_value_v/avNum) #averages vis
        timesReduced.append(new_value_t/avNum) #averages times
        visUncertReduced.append(new_value_u/avNum)
    
    #Calculate phase from visibility average
    phsAv = []
    phsUncertAv = []
    for l in range (0,len(visAvReduced)):
            # calculate phase
            phsAv.append((cmath.phase(visAvReduced[l])))
            # Calculate error on phase, (outlined pg 32 of lab book)
            phsUncertAv.append(visUncertReduced[l].real*pow((visAvReduced[l].real*visAvReduced[l].real + visAvReduced[l].imag*visAvReduced[l].imag),-0.5))
    #turn from list into numpy array
    phsAv=np.array(phsAv)
    phsUncertAv=np.array(phsUncertAv)    

    #Plot these if you want - now properly averaged!!!
    plt.figure()
    #plt.scatter(times, phsAv)
    plt.errorbar(timesReduced, phsAv, phsUncertAv, fmt='o', ecolor='g')
    plt.show()
    

    #Output the data in a form to be used by the GP programme
    #Stitch together data arrays and transpose to columns
    data = np.array([timesReduced, phsAv, phsUncertAv])
    data = data.T
    
    #Open a .txt file to write to 
    #format.(i) makes a different file for each baseline thats being looped over
    with open("{}datafile.txt".format(i), 'wb+') as datafile_id:
    #Write the data, formatted and separated by a comma
        np.savetxt(datafile_id, data, fmt=['%.2f','%.5f','%.5f'], delimiter=',')

