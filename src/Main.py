import Arne
import Chris
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

# This function executes fuzzy dbscan of ... on a given set of data points.
# The algorithm is based on Fuzzy Core DBScan Clustering Algorithm by
# Gloria Bordogna and Dino Ienco. 
#
# It takes a numpy matrix of data points (data),
# the epsilon distance and an interval between minPtsMin
# and minPtsMax that shows if a point has enough points
# in it's epsilon neighborhood to be a core point.
# showClustering is a boolean that tells the program
# to visualize the clustering or not.
def main(data, eps, minPtsMin, minPtsMax, showClustering = True):
    # Compute distances of data points
    distances = Chris.computeDistances(data)
    
    # Create a clustering using fuzzy dbscan
    clustering = Arne.fuzzyDBSCAN(data, distances, eps, minPtsMin, minPtsMax)
    
    # Show a nice picture if desired
    if showClustering:
        Plotter.visualizeClustering(data, clustering, eps, minPtsMin, minPtsMax)

# Start program for test
main(data,epsilon,minPtsMin,minPtsMax)