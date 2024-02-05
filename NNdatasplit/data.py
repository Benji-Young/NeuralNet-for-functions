import numpy as np
import pathlib
import matplotlib.pyplot as plt
import math

function = lambda x : (np.sin(x / 5) * 50) + 50

def get_mnist():

    l = 10000

    points = np.random.uniform(0, 100, size=(l,2)).astype(int)
    labels = np.array([0] * len(points))
    for i in range(len(points)):
        if function(points[i][0]) >= points[i][1]:
            labels[i] = 1
        else:
            labels[i] = 0

    #print(len(points[:, 0]))
    #print(len(points[:, 1]))
    #print(0 in labels)
    #plt.scatter([points[:, 0]], [points[:, 1]])
    #plt.show()
    return points, labels
