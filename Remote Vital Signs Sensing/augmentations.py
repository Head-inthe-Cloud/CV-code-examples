import h5py
import hdf5storage
import cv2
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch


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


# Use pixel value thresholding to find background, epsilon is adjusted manually
def bg_aug_simple(image, background, epsilon=150):
    x1 = 0
    y1 = 0
    x2 = image.shape[1] - 1
    y2 = int(np.floor(image.shape[0] / 2))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # search top-left, top-right, mid-left, and mid-right to ensure background color
    c_dict = {}
    colors = []
    for i, (x, y) in enumerate([[x1, y1], [x2, y1], [x1, y2], [x2, y2]]):
        colors.append(image[y][x])
        if i not in c_dict:
            c_dict[i] = 1
        else:
            c_dict[i] += 1

    c_bg = colors[max(c_dict, key=c_dict.get)]
    u_bg = c_bg + epsilon
    l_bg = c_bg - epsilon
    print(c_bg, u_bg, l_bg)
    # I used the minimum value in the picture as black, could be a very small value as well
    mask = cv2.inRange(image, l_bg, u_bg)
    print("mask max:{}, min:{}".format(np.amax(mask), np.amin(mask)))
    res = cv2.bitwise_and(image, image, mask=mask)

    result = image - res
    result = np.where(result == 0, background, result)
    result = cv2.cvtColor(result.astype('float32'), cv2.COLOR_HSV2RGB)
    return result


def bg_aug_matte(image, background, matte):
    return np.array(image * matte, dtype='uint8') + np.array(background * (1 - matte), dtype='uint8')


def lighting_aug_gamma(image, gamma):
    gamma_table = np.array([np.power(x / 255.0, gamma) * 255.0 for x in range(256)]).clip(0, 255).astype('uint8')
    # cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
    image = (image - np.amin(image)) / (np.amax(image) - np.amin(image)) * 255
    image = image.astype('uint8')
    result = cv2.LUT(image, gamma_table)
    result = result / 255
    return result  # np.clip(result, 0, 255) # No change from simply result


def lighting_aug_alpha(image, alpha, beta=150):
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    result = result / 255
    return result


if __name__ == '__main__':
    # For lighting augmentation and threshold background augmentation
    input_path = "/home/patrick/CVProjects/MTTS-CAN/data/UBFC_Raw/results/original/P1.mat"
    # For background augmentation with matte
    matte_path = "/home/patrick/CVProjects/MTTS-CAN/results/alpha_matte/P1/P1_0000000.png"
    videoFilePath = "/home/patrick/CVProjects/MTTS-CAN/data/UBFC_Raw/dataset/subject1/vid.avi"

    motion, appearance, label, _ = extract(input_path)
    img = appearance[0]
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    plt.imshow(img)
    plt.show()

    result = lighting_aug_gamma(img, 0.5)
    plt.imshow(result)
    plt.show()

    result = lighting_aug_alpha(img, 190, 50)
    plt.imshow(result)
    plt.show()

    bg_color = np.array([0, 0, 0])
    background = np.ones(img.shape) * bg_color

    result = bg_aug_simple(img, background, 20)
    plt.imshow(result)
    plt.show()

    vidObj = cv2.VideoCapture(videoFilePath)
    success, img = vidObj.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bg_color = np.array([0, 0, 0])
    background = np.ones(img.shape) * bg_color
    matte = cv2.imread(matte_path) / 255
    result = bg_aug_matte(img, background, matte)
    plt.imshow(result)
    plt.show()

