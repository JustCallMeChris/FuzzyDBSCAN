import FuzzyDBSCAN
import Plotter
import numpy as np

# Function to read data out of arff file
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
    
    arrayOfPoints = np.array(arrayOfPoints)
    
    # Array of array, each array contains the coordinates of the point, like [[5.1, 3.5, 1.4, 0.2], .., [5.0, 3.6, 1.4, 0.2]] 
    return arrayOfPoints

# This function executes fuzzy dbscan on a given set of data points.
# The algorithm is based on Fuzzy Core DBScan Clustering Algorithm by
# Gloria Bordogna and Dino Ienco. Based on means we adapted it as far
# as necessary to be intuitively what they might have meant in their paper
# with all the errors and lacks of information in it.
# 
# The resulting clustering is printed to standard out.
#
# Parameters are:
# eps:            epsilon distance
# minPtsMin:      minimum amount of points to be in the neighborhood of a
#                 data point p for p to be recognized as a core point.
# mintPtsMax:     maximum amount a points in the neighborhood of a data point
#                 which leads to maximum membership degree of 1 for points with
#                 at least minPtsMax neighbors. This parameter helps to recognize
#                 more degrees of density. Thats's why it is recommended to use
#                 big values.
# data:           String containing the path to an available arff file
#                 OR
#                 a numpy.ndarray with data points as rows and columns as attributes.
# createImage:    Boolean that tells the program to create an image of the resulting
#                 clustering in the operation system's temporary directory.
def main(eps, minPtsMin, minPtsMax, data, createImage = True):
    
    # If an arff file is given, read file and convert data into numpy.ndarray
    if isinstance(data, basestring):
        arfff = open(data,"r")
        data = arffFileToArrayOfPoints(arfff)
    
    # If no data is given, return.
    if data.shape[0] == 0:
        print "No data! Return!"
        return
    
    # Create a clustering using fuzzy dbscan
    clustering = FuzzyDBSCAN.fuzzyDBSCAN(data, eps, minPtsMin, minPtsMax)
    
    # Print clustering on standard out
    for clusterid in clustering:
        print clusterid
    
    # Show a nice picture if desired
    if createImage:
        Plotter.visualizeClustering(data, clustering, eps, minPtsMin, minPtsMax)


############ TEST INFO ############
# Load test data
#data = np.loadtxt(open("../spirals.csv","r"),delimiter=";")
#data = np.loadtxt(open("../moons.csv","r"),delimiter=";")
#data = "../iris.arff"
#data = "../empty.arff"
#data = ""
# Test parameters for fuzzy dbscan clustering
#minPtsMin = 4
#minPtsMax = 1000
#epsilon = 2.0
# Create clustering
#main(epsilon, minPtsMin,minPtsMax, data)