# This function computes the euclidean distance of a matrix of data points.
# Input is the numpy matrix of data points.
# Output is a matrix of (data point x data point), that is filled with numbers/distances.


import numpy as np

def arffFileToArrayOfPoints(arffFile):
    
    dataString = arffFile.read()
    # Splits read file line by line into array of separate strings
    dataArrayOfStrings = dataString[dataString.index('@DATA')+6:].split('\n')
    # Reads dimension  
    dimension = int(dataString[dataString.index('dimensions:'):].split('\n')[0][-1])
    
    listOfDataPoints = []
    
    for i in range(len(dataArrayOfStrings)):
        # Splits strings (which contains values) into separate values (still as string)
        pointsAsStrings = dataArrayOfStrings[i].split(',')
        # Removes Ground-Truth-Label
        pointsAsStrings = pointsAsStrings[:-1]
        if pointsAsStrings: #if not not pointsAsStrings:
            # Convert string "values" into float value
            listOfDataPoints.extend(map(float, pointsAsStrings))  
    # Splits array into array of array, each array contains the coordinates of the point
    arrayOfPoints = [listOfDataPoints[i:i+dimension] for i in range(0, len(listOfDataPoints), dimension)]
    
    # Array of array, each array contains the coordinates of the point, like [[5.1, 3.5, 1.4, 0.2], .., [5.0, 3.6, 1.4, 0.2]] 
    return arrayOfPoints


def computeDistances(data):
    
    arrayOfPoints = data #arffFileToArrayOfPoints(arffFile)
    lenArrayOfPoints = len(arrayOfPoints)
    dimension = len(arrayOfPoints[0])
    
    distanceMatrix = []
    
    # Rows
    for i in range(lenArrayOfPoints): 
        
        distanceCollector = [[]]
        # Columns
        for j in range(lenArrayOfPoints): 
            if i >= j: 
                # Fills lower triangular matrix with 0
                distanceCollector[0].extend([0])
            else:
                euclideanDistanceAddition = 0
                
                # Computes euclidean distance
                for k in range(dimension):
                    euclideanDistanceAddition = euclideanDistanceAddition + (arrayOfPoints[i][k]-arrayOfPoints[j][k])**2
                euclideanDistance = euclideanDistanceAddition**(1/2.0)   
                distanceCollector[0].extend([euclideanDistance])
        
        # Adds row to array of distance matrix        
        distanceMatrix.extend(distanceCollector)
    #Distance Matrix as NumpyArray    
    distanceMatrix = np.array(distanceMatrix)

    return distanceMatrix


