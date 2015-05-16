import numpy as np

# This function executes a fuzzy dbscan algorithm.
# Input is a numpy matrix of data points (data), a matrix of distances (distances)
# the number of data points a core point has to have at least and at most.
# Output is a vector of clusters for every data point.
def fuzzyDBSCAN(data, distances, eps, minPtsMin, minPtsMax):
    print "Arne's Function"
    
    clustering = np.array([1,2,3,4,5])
    return clustering