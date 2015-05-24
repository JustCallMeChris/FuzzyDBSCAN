import Arne
import Chris
import Plotter
import numpy as np



# TODO: Check, what function interface we have to supply!



# Load test data
#data = np.loadtxt(open("../spirals.csv","r"),delimiter=";")
data = open("../iris.arff","r")

# Parameters for fuzzy dbscan clustering
minPtsMin = 5
minPtsMax = 10
epsilon = 2.0

# This function executes fuzzy dbscan of ... on a given set of data points.
# The algorithm is based on Fuzzy Core DBScan Clustering Algorithm by
# Gloria Bordogna and Dino Ienco. 
#
# It takes a numpy matrix of data points (data),
# the epsilon distance and an interval of points that shows if a point has
# enough points in it's epsilon neighborhood.  
def main(data, eps, minPtsMin, minPtsMax):
    # Compute distances of data points
    distances = Chris.computeDistances(data)
    
    # Create a clustering using fuzzy dbscan
    #clustering = Arne.fuzzyDBSCAN(data, distances, eps, minPtsMin, minPtsMax)
    
    # Show a nice picture :)
    #Plotter.visualizeClustering(data, clustering)

# Start program
main(data,epsilon,minPtsMin,minPtsMax)