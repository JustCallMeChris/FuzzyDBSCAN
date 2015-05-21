# Show a nice picture ;)
# Input: Numpy matrix (not yet) of data points (data) and
# list of cluster ids for the given data points (clustering).

import matplotlib.pyplot as plt
import numpy as np


def visualizeClustering(data, clustering):
    
    data = np.array(data)
    clustering = np.array(clustering)
    
    unique_labels = set(clustering) # eliminates multiple values
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels))) # color for each cluster 
    
    for k, color in zip(unique_labels, colors): # colored dot for dot
        if k == -1:
            color = 'k'   # noise is shown as black dots

        class_member_mask = (clustering == k)   #true/false matrix to know which dots belongs to cluster

        xy = data[class_member_mask] #points of the cluster or noise    
              
        
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)   #form, color, bordercolor
    #plt.title('Clustering with DBSCAN')
    plt.show()