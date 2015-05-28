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
    
    # Matrix to store membership degrees of points.
    # Initiated to -1 just to reserve an index in the matrix.
    memberships = [[-1] for i in range(numPoints)]
    
    # Array to store if a point is already visited.
    # Visited indicates we already computed the
    # eps-neighborhood once for core points.
    visited = [0] * numPoints
    
    for i in range(numPoints):
        # If the current data point was already visited before,
        # stop here.
        if visited[i]:
            continue
        
        # Compute eps-neighborhood of current data point
        neighbors = computeNeighbors(distances, i, eps)
        
        # If this data point is a core point, treat it appropriately.
        if len(neighbors) > minPtsMin:
            # Mark current data point as visited
            visited[i] = True
            
            # Increment cluster id
            currentCluster += 1
            
            # Add a column to memberships if necessary
            # Might be done more efficiently.
            if currentCluster > 0:
                for row in memberships:
                    row.append(-1)
            
            # Grow this cluster
            expandFuzzyCluster(i, neighbors, eps, minPtsMin, minPtsMax, visited, memberships, distances, currentCluster)
        
    # Compute clustering out of membership matrix
    # -1 is noise, everything else is a cluster index
    clustering = []
    for i in range(numPoints):
        cluster = -1
        maxMembership = -1
        for j in range(currentCluster+1):
            currentMembership = memberships[i][j]
            if currentMembership > maxMembership:
                cluster = j
                maxMembership = currentMembership
        clustering.append(cluster)
    
    return clustering

# Function to compute the eps-neighborhood of
# a data point as a set of indizes.
# distances - distance matrix (upper triangular matrix)
# point - index of data point in distance matrix
# eps - epsilon for epsilon neighborhood
def computeNeighbors(distances, point, eps):
    neighbors = set()
    
    # Look at first part of distances
    for i in range(point):
        if distances[i][point] <= eps:
            neighbors.add(i)
    
    # Look at second part of distances
    numPoints = distances.shape[1]
    # We insert the point itself as it's own neighbor.
    for i in range(point,numPoints):
        if distances[point][i] <= eps:
            neighbors.add(i)
    
    return neighbors

#
# This function grows a cluster such that every data point of the cluster currentCluster will be found.
#
def expandFuzzyCluster(point, neighbors, eps, minPtsMin, minPtsMax, visited, memberships, distances, currentCluster):
    # set of border points of this cluster
    borderPoints = set()
    # Set of core points of this cluster
    corePoints = set()
    # Add data point to the current cluster with fuzzy membership degree
    memberships[point][currentCluster] = computeMembershipDegree(len(neighbors), minPtsMin, minPtsMax)
    # Add core point to set of core points
    corePoints.add(point)
    
    # As long as neighbors is not empty
    while neighbors:
        i = neighbors.pop()
        # If this neighbor is already visited
        if not visited[i] and not (i in borderPoints):
            # Compute neighbors of current neighbor
            neighbors2 = computeNeighbors(distances, i, eps)
            
            # Core point
            if len(neighbors2) > minPtsMin:
                # Mark current neighbor as visited
                visited[i] = True
                # Add core point to set of core points
                corePoints.add(i)
                # Take neighbors into consideration
                neighbors = neighbors.union(neighbors2)
                # Assign membership degree to this core point
                memberships[i][currentCluster] = computeMembershipDegree(len(neighbors2), minPtsMin, minPtsMax)
            # Border point
            else:
                # Take care: Don't set this point to be visited!
                borderPoints.add(i)
    
    # Compute membership degrees of this cluster's border points
    # to introduce the desired fuzzy aspect.
    while borderPoints:
        i = borderPoints.pop()
        # Compute neighbors of this data point.
        # This might happen more than once for border points.
        neighbors2 = computeNeighbors(distances, i, eps)
        # Which neighbors are core points of this cluster?
        coreNeighbors = neighbors2.intersection(corePoints)
        # Which core point has the biggest membership degree?
        biggestMembership = -1
        while coreNeighbors:
            j = coreNeighbors.pop()
            currentMembership = memberships[j][currentCluster]
            if biggestMembership < currentMembership:
                biggestMembership = currentMembership
        
        # Set membership degree of current border point to
        # the maximum membership degree of its core neighbors
        # of this cluster.
        memberships[i][currentCluster] = biggestMembership
    
    return
    
# Function to calculate fuzzy membership degrees.
# numNeighbors is the number of neighbors of a data point. 
def computeMembershipDegree(numNeighbors, minPtsMin, minPtsMax):
    if numNeighbors >= minPtsMax:
        return 1
    if numNeighbors > minPtsMin:
        return float(numNeighbors - minPtsMin) / float(minPtsMax - minPtsMin)
    # This is currently not really needed for this function
    # for it is only called for core points.
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
    #print distanceMatrix  

    return distanceMatrix
