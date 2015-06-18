import numpy as np

# This function executes a fuzzy dbscan algorithm.
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
#                 a numpy.ndarray with data points as rows and columns as attributes
#
# Returns a vector of cluster ids, one cluster id for each data point.
def fuzzyDBSCAN(data, eps, minPtsMin, minPtsMax):
    
    # Compute distances of data points
    distances = computeDistances(data)
    
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
    visited = [False] * numPoints
    
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
            expandFuzzyCluster(i, neighbors, eps, minPtsMin, minPtsMax, visited, memberships, distances, currentCluster, data)
        
    # Compute crisp clustering out of membership matrix
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

# Function to compute the eps-neighborhood of a data point as a set of indizes.
#
# Parameters are:
# distances:    numpy.ndarray that is an upper triangular matrix with diagonal 0-entries.
# point:        Index in distance matrix of data point to compute the neighborhood for.
# eps:          Value to define distance of epsilon neighborhood epsilon.
#
# Returns set of neighbor points as indizes into distance matrix.
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
# Parameters are:
# point:            First processed core point of this cluster
# neighbors:        Epsilon neighborhood of point (this is a set)
# eps:              Value to define distance of epsilon neighborhood epsilon.
# minPtsMin:        minimum amount of points to be in the neighborhood of a
#                   data point p for p to be recognized as a core point.
# mintPtsMax:       maximum amount a points in the neighborhood of a data point
#                   which leads to maximum membership degree of 1 for points with
#                   at least minPtsMax neighbors. This parameter helps to recognize
#                   more degrees of density. Thats's why it is recommended to use
#                   big values.
# visited:          Array of flags to show if the the epsilon neighborhood has already
#                   been computed for each of the data points.
# memberships:      Matrix to store membership degrees of points.
# distances:        numpy.ndarray that is an upper triangular matrix with diagonal 0-entries.
# currentCluster:   Index of the currently processed cluster
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
        # If this neighbor is not already visited
        # and not a border point.
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
#
# Parameters are:
# numNeighbors:    Number of neighbors of a data point.
# minPtsMin:       minimum amount of points to be in the neighborhood of a
#                  data point p for p to be recognized as a core point.
# mintPtsMax:      maximum amount a points in the neighborhood of a data point
#                  which leads to maximum membership degree of 1 for points with
#                  at least minPtsMax neighbors. This parameter helps to recognize
#                  more degrees of density. Thats's why it is recommended to use
#                  big values.
def computeMembershipDegree(numNeighbors, minPtsMin, minPtsMax):
    if numNeighbors >= minPtsMax:
        return 1
    if numNeighbors > minPtsMin:
        return float(numNeighbors - minPtsMin) / float(minPtsMax - minPtsMin)
    # This is currently not really needed for this function
    # for it is only called for core points.
    if numNeighbors <= minPtsMin:
        return 0;

# This function computes the Euclidean distance of a matrix of data points.
# Parameters are:
# data:        numpy.ndarray of data points.
#
# Returns an upper triangular matrix (with diagonal 0-values),
# that is filled with numbers/distances.
def computeDistances(data):

    lenArrayOfPoints = len(data)
    dimension = len(data[0])
    
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
                    euclideanDistanceAddition = euclideanDistanceAddition + (data[i][k]-data[j][k])**2
                    
                euclideanDistance = euclideanDistanceAddition**(1/2.0)   
                distanceCollector[0].extend([euclideanDistance])
        
        # Adds row to array of distance matrix        
        distanceMatrix.extend(distanceCollector)
    # Distance Matrix as NumpyArray  
    distanceMatrix = np.array(distanceMatrix)  

    return distanceMatrix
