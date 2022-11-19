import numpy as np
import math

EPSILON = 10e-6

# [x, y] -> [angle, range]
def toar(point):
    if len(np.shape(point)) == 2:
        return [[np.arctan2(p[1], p[0]), math.sqrt(pow(p[0], 2) + pow(p[1], 2))] for p in point]
    else:
        return [np.arctan2(point[1], point[0]), math.sqrt(pow(point[0], 2) + pow(point[1], 2))]


# [angle, range] -> [x, y]
def toxy(point):
    if len(np.shape(point)) == 2:
        return [[p[1] * math.cos(p[0]), p[1] * math.sin(p[0])] for p in point]
    else:
        return [point[1] * math.cos(point[0]), point[1] * math.sin(point[0])]


# Normalize data to 0~1 range
def normalize(data):
    if type(data) != np.ndarray:
        data = np.array(data)
    max_val = np.max(data)
    min_val = np.min(data)
    return (data - min_val) / (max_val - min_val)
