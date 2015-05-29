# Show a nice picture ;)
# Input: Numpy array of data points (data) and
# List of cluster labels for the given data points (clustering).

import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile as tf


def visualizeClustering(data, clustering, eps, minPoints, maxPoints):
    
    #data = np.array(data)    
    #print type(data)

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
              
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)   #form, color, bordercolor
            
    plt.title('Fuzzy DBSCAN Clustering')
    plt.xlabel('Eps: '+str(eps)+'    minPtsMin: '+str(minPoints)+'    minPtsMax: '+str(maxPoints))
    #plt.draw()
    
    # Saves the image
    imageName = "fuzzyDBSCAN.png"
    tempPath = tf.mkdtemp()
    plt.savefig(tempPath+os.sep+imageName, dpi=100)
    print("# Image: "+tempPath + os.sep + imageName)
    
    # Shows figure in popup-window
    #plt.show()