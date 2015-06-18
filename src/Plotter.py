import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile as tf

# This function plots an image for given set of data points. 
# The clustering is printed to standard out.
#
# Parameters are:
# data:           a numpy.ndarray with data points as rows and columns as attributes.
# clustering:     list of cluster labels for given data points
# eps:            epsilon distance
# minPoints:      minimum amount of points to be in the neighborhood of a
#                 data point p for p to be recognized as a core point.
# maxPoints:      maximum amount a points in the neighborhood of a data point
#                 which leads to maximum membership degree of 1 for points with
#                 at least minPtsMax neighbors. This parameter helps to recognize
#                 more degrees of density. Thats's why it is recommended to use
#                 big values.


def visualizeClustering(data, clustering, eps, minPoints, maxPoints):
    # Eliminates multiple values
    unique_labels =set(np.unique(clustering))

    # Color for each cluster
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # Colored dot for dot
    for k, color in zip(unique_labels, colors):
        if k == -1:
            # Noise is shown as black dots
            color = 'k'

        # True/False matrix to know which dots belongs to cluster
        class_member_mask = (clustering == k)
        # Points of the cluster or noise
        xy = data[class_member_mask]
        # Plots points, last parameters means form, color and bordercolor
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)

    plt.title('Fuzzy DBSCAN Clustering')
    plt.xlabel('Eps: '+str(eps)+'    minPtsMin: '+str(minPoints)+'    minPtsMax: '+str(maxPoints))

    # Saves the image in temporary folder
    imageName = "fuzzyDBSCAN.png"
    tempPath = tf.mkdtemp()
    plt.savefig(tempPath+os.sep+imageName, dpi=100)
    print("# Image: "+tempPath + os.sep + imageName)
