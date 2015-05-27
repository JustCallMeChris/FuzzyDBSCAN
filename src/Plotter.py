# Show a nice picture ;)
# Input: Numpy matrix (not yet) of data points (data) and
# list of cluster ids for the given data points (clustering).

import matplotlib.pyplot as plt
import numpy as np
import os

#TODO Ausgabe PNG + Link zum File

def visualizeClustering(data, clustering, eps, minPoints, maxPoints):
    
    data = np.array(data)
    #print clustering
    
    unique_labels =set(np.unique(clustering)) # eliminates multiple values
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels))) # color for each cluster 
    
    for k, color in zip(unique_labels, colors): # colored dot for dot
        if k == -1:
            color = 'k'   # noise is shown as black dots

        class_member_mask = (clustering == k)   #true/false matrix to know which dots belongs to cluster

        xy = data[class_member_mask] #points of the cluster or noise    
              
        
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)   #form, color, bordercolor
    plt.title('Fuzzy DBSCAN Clustering')
    plt.xlabel('Eps: '+str(eps)+'    minPtsMin: '+str(minPoints)+'    minPtsMax: '+str(maxPoints))
    #plt.draw()
    
    # Saves the image
    imageName = "fuzzyDBSCAN.png"
    plt.savefig(imageName, dpi=100)
    
    # Prints fullpath of the file
    full_path = os.path.realpath(__file__)
    path, file = os.path.split(full_path)
    print(path+"/"+imageName) 
    
    # Shows figure in separate popup-window
    #plt.show()





    
    
    