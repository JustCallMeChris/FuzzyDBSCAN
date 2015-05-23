import numpy as np

# This function executes a fuzzy dbscan algorithm.
# Input is a numpy matrix of data points (data), a matrix of distances (distances)
# the number of data points a core point has to have at least and at most.
# Output is a vector of clusters for every data point.
def fuzzyDBSCAN(data, distances, eps, minPtsMin, minPtsMax):
    ################################################
    ###### Step 1: Compute membership degrees ######
    ################################################
    
    # Dimensions of data matrix
    numPoints = data.shape[0]
    
    # Current cluster
    # This is used to add columns to memberships matrix.
    currentCluster = -1
    
    # Structure to store membership degrees.
    # This is a matrix with numPoints rows and
    # numClusters cols.
    # Entries are floatables. An entry at row 1 and col 2
    # represents membership degree of data point 2 for cluster 3
    # if that entry is bigger than 0. If this entry is zero
    # this data point is a border point of cluster 3. 
    # This matrix is initiated with one column and -1 for every entry.
    memberships = np.empty((numPoints,1), dtype=float)
    memberships[:] = -1
    
    # Array to store if a point is already visited.
    # Visited indicates we already computed the eps-neighborhood once.
    visited = np.zeros((numPoints,1), dtype=bool)
    
    for i in range(0,numPoints):
        # If the current data point was already visited before,
        # stop here.
        if visited[i]:
            continue
        # Mark current data point as visited
        visited[i] = True
        
        # Compute eps-neighborhood of current data point
        neighbors = computeNeighbors(distances, i, eps) 
        
        # If not enough neighors: This data point is noise
        # which is already stored in membership matrix!
        # If this data point is a core point, treat it appropriately.
        if len(neighbors) > minPtsMin:
            # Increment cluster id
            currentCluster += 1
            # Add a column to memberships if necessary
            if currentCluster <> 0:
                newShape = memberships.shape
                newShape[1] += 1
                newMemberships = np.empty(newShape, dtype=float)
                newMemberships[:] = -1
                newMemberships[:,:-1] = memberships
                memberships = newMemberships
            expandFuzzyCluster(i, neighbors, eps, minPtsMin, minPtsMax, visited, memberships, distances)
    
    ################################################
    ######     Step 2: Compute clustering     ######
    ######     out of membership degrees      ######
    ################################################
    # Data structure of the clustering
    # Entries are noise if -1 or
    # cluster indizes otherwise.
    clustering = np.empty((numPoints,1))
    
    # Find maximum membership degrees for every data point
    # and appropriately assign cluster ids to the data points.
    for i in range(0,numPoints):
        maxDegreeOfPoint = memberships[i][0]
        maxDegreeCluster = 0
        for j in range(1,currentCluster+1):
            currentDegree = memberships[i][j]
            if maxDegreeOfPoint < currentDegree:
                maxDegreeOfPoint = currentDegree
                maxDegreeCluster = j
        
        # Assign cluster to current data point
        clustering[i] = maxDegreeCluster

    # Print clustering
    for i in range(0, numPoints):
        print i , ", " , clustering[i]
    
    return clustering

# Function to compute the eps-neighborhood of
# a data point as a set of indizes.
# distances - distance matrix
# point - index of data point in distance matrix
# eps - epsilon for epsilon neighborhood
def computeNeighbors(distances, point, eps):
    neighbors = set()
    numPoints = distances.shape[0]
    for i in range(0,numPoints):
        if distances[point][i] <= eps:
            neighbors.add(i)
    return neighbors

#
# TODO: Write some documentation
#
def expandFuzzyCluster(point, neighbors, eps, minPtsMin, minPtsMax, visited, memberships, distances):
    # Index of current cluster
    currentCluster = memberships.shape[1]-1
    
    # Add data point to the current cluster with fuzzy membership degree
    memberships[point][currentCluster] = computeMembershipDegree(len(neighbors), minPtsMin, minPtsMax)
    
    for i in neighbors:
        # If this neighbor is already visited
        if visited[i]:
            continue
        
        # Mark current neighbor as visited
        visited[i] = True
        
        # Compute neighbors of current neighbor
        neighbors2 = computeNeighbors(distances, i, eps)

        if len(neighbors2) > minPtsMin:
            neighbors.add_all(neighbors2)
            memberships[i][currentCluster] = computeMembershipDegree(len(neighbors2), minPtsMin, minPtsMax)
        else:
            memberships[i][currentCluster] = 0
    
    return
    

# Function to calculate fuzzy membership degrees.
# numNeighbors is the number of neighbors of a data point. 
def computeMembershipDegree(numNeighbors, minPtsMin, minPtsMax):
    if numNeighbors >= minPtsMax:
        return 1
    if numNeighbors > minPtsMin:
        return float(numNeighbors - minPtsMin) / float(minPtsMax - minPtsMin)
    # This is defined in theory but should never be processed!
    if numNeighbors <= minPtsMin:
        return 0;
