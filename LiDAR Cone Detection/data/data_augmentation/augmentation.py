import math
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from utils import read_file, write_file, plot


A_RANGE = math.pi / 180 * 2.2 / 2  # The angle of perception = A_RANGE x 2
IN_NAME = "lidar_data_10m.csv"
OUT_NAME = IN_NAME + '_augmented'

def augment():

    datasets = read_file(IN_NAME)
    plot(datasets, num_show=1)

    print(np.shape(datasets))

    new_dataset = []
    for i in range(len(datasets)):
        angles = datasets[i][0]
        ranges = datasets[i][1]

        # -3.14 ~ 3.14 -> 0 ~ 360

        for i in range(len(angles)):
            angles[i] -= math.pi / 2


        new_angles = []
        new_ranges = []
        new_intensity = []

        upper_bound = math.pi / 2 - A_RANGE
        lower_bound = -math.pi / 2 * 3 + A_RANGE

        for i in range(len(angles)):
            if -math.pi / 2 * 3 <= angles[i] <= lower_bound or upper_bound <= angles[i] <= math.pi / 2:
                new_angles.append(angles[i])
                new_ranges.append(ranges[i])
                new_intensity.append(1008.0)

        # new_dataset.append([new_angles, [dict[angle] for angle in new_angles], datasets[0][2]])
        new_dataset.append([new_angles, new_ranges, new_intensity])

    plot(new_dataset, num_show=1)

    write_file(new_dataset, OUT_NAME)
    for i in range(np.shape(new_dataset)[0]):
        print(np.shape(new_dataset[0]))

if __name__ == "__main__":
    # augment()
    plot(read_file(IN_NAME), num_show=10)





