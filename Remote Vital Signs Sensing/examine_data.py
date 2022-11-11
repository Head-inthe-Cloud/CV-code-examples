import h5py
import hdf5storage
import cv2
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

INPUT_PATH = "/home/patrick/CVProjects/MTTS-CAN/data/results/P1/P1C1.mat"

def examine_file(data_path):
    with h5py.File(data_path, 'r') as f:
        data = torch.tensor(np.array(f['dXsub']))
        print("Before Transformation: ", data.shape)
        data = data.permute(3, 2, 1, 0)[..., 3:6]
        print("After Transoformation: ", data.shape)
        data_flat = torch.flatten(data[0])
        data_max, data_min = max(data_flat), min(data_flat)
        print("Data Range: ", data_max, data_min)
        img = data.detach().cpu().numpy()[0]
        print(img.shape)
        cv2.imshow('img', img)
        cv2.waitKey(0)

        dysub = np.array(f['dysub'])
        print("dysub shape", dysub.shape, "Type: ", type(dysub[0][0]))
        print("dysub example", dysub[0][:20])


if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', type=str, help='path of input .mat file')
    args = parser.parse_args()

    examine_file(args.input_path)

