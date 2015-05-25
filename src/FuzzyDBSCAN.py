import numpy as np

# This function executes a fuzzy dbscan algorithm.
# Input is a numpy matrix of data points (data), a matrix of distances (distances)
# the number of data points a core point has to have at least and at most.
# Output is a vector of clusters for every data point.
def fuzzyDBSCAN(data, distances, eps, minPtsMin, minPtsMax):
    # Dimensions of data matrix
    numPoints = data.shape[0]
    
    # Current cluster
    # This is used to add columns to memberships matrix.
    currentCluster = -1
    
    # Data structure of the clustering
    # Entries are noise if -1 or
    # cluster indizes otherwise.
    # Initiated as array of noise points.
    clustering = [-1] * numPoints
    
    # Array to store "classes" of points noise, border or core.
    # If noise an entry will be -1, if border 0,
    # if core it will be in (0,1] to show how strong a data point
    # belongs to a cluster.
    # Initiated as array of noise points.
    memberships = [-1] * numPoints
    
    # Array to store if a point is already visited.
    # Visited indicates we already computed the eps-neighborhood once.
    visited = [0] * numPoints
    
    for i in range(0,numPoints):
        # If the current data point was already visited before,
        # stop here.
        if visited[i]:
            continue
        # Mark current data point as visited
        visited[i] = True
        
        # Compute eps-neighborhood of current data point
        neighbors = computeNeighbors(distances, i, eps) 
        
        # If this data point is a core point, treat it appropriately.
        # If not enough neighbors: This data point is noise
        # which is already stored in membership array!
        if len(neighbors) > minPtsMin:
            # Increment cluster id
            currentCluster += 1
            # Grow this cluster
            expandFuzzyCluster(i, neighbors, eps, minPtsMin, minPtsMax, visited, memberships, distances, clustering, currentCluster)
    
    return clustering

# Function to compute the eps-neighborhood of
# a data point as a set of indizes.
# distances - distance matrix
# point - index of data point in distance matrix
# eps - epsilon for epsilon neighborhood
def computeNeighbors(distances, point, eps):
    neighbors = set()
    numPoints = distances.shape[1]
    for i in range(0,numPoints):
        if distances[point][i] <= eps:
            neighbors.add(i)
    return neighbors

#
# This function grows a cluster such that every data point of the cluster currentCluster will be found.
#
def expandFuzzyCluster(point, neighbors, eps, minPtsMin, minPtsMax, visited, memberships, distances, clustering, currentCluster):
    # Add data point to the current cluster with fuzzy membership degree
    memberships[point] = computeMembershipDegree(len(neighbors), minPtsMin, minPtsMax)
    clustering[point] = currentCluster
    
    # As long as neighbors is not empty
    while neighbors:
        i = neighbors.pop()
        # If this neighbor is already visited
        if visited[i]:
            continue
        
        # Mark current neighbor as visited
        visited[i] = True
        
        # Compute neighbors of current neighbor
        neighbors2 = computeNeighbors(distances, i, eps)

        # If this is a core point take it's neighbors into consideration.
        if len(neighbors2) > minPtsMin:
            neighbors = neighbors.union(neighbors2)
        
        # Assign membership degree to this data point (core or border point)
        memberships[i] = computeMembershipDegree(len(neighbors2), minPtsMin, minPtsMax)
        clustering[i] = currentCluster
    
    return
    
# Function to calculate fuzzy membership degrees.
# numNeighbors is the number of neighbors of a data point. 
def computeMembershipDegree(numNeighbors, minPtsMin, minPtsMax):
    if numNeighbors >= minPtsMax:
        return 1
    if numNeighbors > minPtsMin:
        return float(numNeighbors - minPtsMin) / float(minPtsMax - minPtsMin)
    if numNeighbors <= minPtsMin:
        return 0;

# This function computes the euclidean distance of a matrix of data points.
# Input is the numpy matrix of data points.
# Output is a matrix of (data point x data point), that is filled with numbers/distances.
# Can still be optimized, but works for now
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
            if i == j: 
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
