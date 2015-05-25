import FuzzyDBSCAN
import Plotter
import numpy as np

# Load test data
data = np.loadtxt(open("../spirals.csv","r"),delimiter=";")
#data = open("../iris.arff","r")
#data = Chris.arffFileToArrayOfPoints(data)
# Test parameters for fuzzy dbscan clustering
minPtsMin = 4
minPtsMax = 20
epsilon = 2.0

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
    
    # Array of array, each array contains the coordinates of the point, like [[5.1, 3.5, 1.4, 0.2], .., [5.0, 3.6, 1.4, 0.2]] 
    return arrayOfPoints


# This function executes fuzzy dbscan of ... on a given set of data points.
# The algorithm is based on Fuzzy Core DBScan Clustering Algorithm by
# Gloria Bordogna and Dino Ienco. Based on means we adapted it as far
# as necessary to be intuitively what they might have meant in their paper
# with all the errors and lacks of information in it.
#
# It takes a numpy matrix of data points (data),
# the epsilon distance and an interval between minPtsMin
# and minPtsMax that shows if a point has enough points
# in it's epsilon neighborhood to be a core point.
# showClustering is a boolean that tells the program
# to visualize the clustering or not.
def main(data, eps, minPtsMin, minPtsMax, showClustering = True):
    # Compute distances of data points
    distances = FuzzyDBSCAN.computeDistances(data)
    
    # Create a clustering using fuzzy dbscan
    clustering = FuzzyDBSCAN.fuzzyDBSCAN(data, distances, eps, minPtsMin, minPtsMax)
    
    # Show a nice picture if desired
    if showClustering:
        Plotter.visualizeClustering(data, clustering, eps, minPtsMin, minPtsMax)

# Start program for test
main(data,epsilon,minPtsMin,minPtsMax)