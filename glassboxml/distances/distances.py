import numpy as np

def calculate_euclidean(point1, point2):
    if (len(point1) != len(point2)):
        raise Exception("Dimensions of points are not equal!")
    n_features = len(point1)

    return np.sqrt(np.sum(np.square([point1[i] - point2[i] for i in range(len(point1))])))