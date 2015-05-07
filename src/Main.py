import Andreas
import Arne
import Chris

# This function executes fuzzy dbscan of ... on a given set of data points.
# It takes a numpy matrix of data points (data),
# the epsilon distance and an interval of points that shows if a point has
# enough points in it's epsilon neighborhood.  
def main(data, eps, minPtsMin, minPtsMax):
    # Compute distances of data points
    distances = Andreas.computeDistances(data)
    
    # Create a clustering using fuzzy dbscan
    clustering = Arne.fuzzyDBSCAN(data, distances, eps, minPtsMin, minPtsMax)
    
    # Show a nice picture :)
    Chris.visualizeClustering(data, clustering)

main(1,2,3,4)

print "test"