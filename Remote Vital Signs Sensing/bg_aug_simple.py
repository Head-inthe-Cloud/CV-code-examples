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
        # print("Data sample", data[0])
        for i in range(data.shape[0]):
            img = data.detach().cpu().numpy()[i]
            print(img.shape)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('img', img)
            cv2.waitKey(0)

def extract(data_path):
    with h5py.File(data_path, 'r') as f:
        data = torch.tensor(np.array(f['dXsub']))
        print("Input shape: ", data.shape)
        data = data.permute(3, 2, 1, 0).detach().cpu().numpy()
        return data[..., 0:3], data[..., 3:6], np.array(f['dysub']), np.array(f['drsub'])


# Finds the background color using four points
def find_bg(frame, epsilon):
    assert frame.shape[-1] == 3
    x1 = 0
    y1 = 0
    x2 = frame.shape[1] - 1
    y2 = int(np.floor(frame.shape[0] / 2))

    # search top-left, top-right, mid-left, and mid-right to ensure background color
    c_dict = {}
    colors = []
    for i, (x, y) in enumerate([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]):
        colors.append(frame[y][x])
        if i not in c_dict:
            c_dict[i] = 1
        else:
            c_dict[i] += 1

    c_green = colors[max(c_dict, key=c_dict.get)]
    return c_green + epsilon, c_green - epsilon


# input -- tensor (batch, c, h, w)
def show_img(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Showing Image with shape: ", img.shape)
    cv2.imshow("img", img)
    cv2.waitKey(0)

def color_filter(frame, epsilon):
    u_green, l_green = find_bg(frame, epsilon)
    # I used the minimum value in the picture as black, could be a very small value as well
    # val_black = np.min(frame)
    val_black = -10
    frame_black = np.zeros(frame.shape)
    frame_black.fill(val_black)

    mask = cv2.inRange(frame, l_green, u_green)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    f = frame - res
    f = np.where(f == 0, frame_black, f)

    dim = (frame.shape[1] * 4, frame.shape[0] * 4)
    frame = cv2.resize(frame, dim)
    f = cv2.resize(f, dim)
    cv2.imshow('frame', frame)
    cv2.imshow("mask", f)



def test_colors(img):
    cv2.imshow("test_colors", img)
    epsilon = 0.61
    while True:
        k = chr(cv2.waitKey(0))
        if k == 'w':
            epsilon += 0.01
        elif k == 's':
            epsilon -= 0.01
        elif k == 'x':
            cv2.destroyAllWindows()
            print("The epsilon is ", epsilon)
            break
        else:
            continue
        color_filter(img, epsilon)


if __name__ == '__main__':

    # examine_file("/home/patrick/CVProjects/MTTS-CAN/data/test/result/P1C1.mat")

    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', type=str, help='path of chunk folders')
    parser.add_argument('-o', '--output-path', type=str, help='path of output folder')
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()

    EPSILON = 1.63 # The EPSILON for raw data is 0.61

    for file_name in os.listdir(args.input_path):
        path = os.path.join(args.input_path, file_name)
        output_path = os.path.join(args.output_path, file_name)

        motion, appearance, dysub, drsub = extract(path)

        # Take the pixel value at (w, h) = (0, 0)

        # # Use this line to find a good epsilon
        # test_colors(cv2.cvtColor(appearance[0], cv2.COLOR_BGR2RGB))


        new_data = np.zeros(appearance.shape)

        for i, frame in enumerate(appearance):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            u_green, l_green = find_bg(frame, EPSILON)
            # I used the minimum value in the picture as black, could be a very small value as well
            val_black = np.amin(frame)
            frame_black = np.zeros(frame.shape)
            frame_black.fill(val_black)

            mask = cv2.inRange(frame, l_green, u_green)
            res = cv2.bitwise_and(frame, frame, mask=mask)

            f = frame - res
            f = np.where(f == 0, frame_black, f)

            f = cv2.cvtColor(f.astype('float32'), cv2.COLOR_RGB2BGR)
            new_data[i] = f


        # write data
        output_data = np.concatenate((motion, new_data), axis=-1)
        dysub = dysub.reshape((1, -1))
        drsub = drsub.reshape((1, -1))
        # output_data = torch.tensor(output_data)
        # output_data = output_data.permute(3, 2, 1, 0).detach().cpu().numpy()

        print('Saving file: ', output_path,
              '\n \t dXsub: ', output_data.shape,   # (N, 72, 72, 6)
              '\n \t dysub: ', dysub.shape,         # (1, N)
              '\n \t drsub: ', drsub.shape)         # (1, N)

        matfiledata = {'dXsub': output_data,
                       'dysub': dysub,
                       'drsub': drsub}

        hdf5storage.write(matfiledata, filename=output_path)
