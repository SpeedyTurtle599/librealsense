import numpy as np

def Center(corners):
    center_array = np.round(np.mean(corners, axis=0))
    return center_array.astype(int)