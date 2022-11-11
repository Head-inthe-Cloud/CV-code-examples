import os
import argparse
import sys

import numpy as np
import cv2
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import time
import scipy.io
import hdf5storage
from augmentations import bg_aug_matte, lighting_aug_gamma, lighting_aug_alpha



# This code is originated from Xin Liu's MTTS-CAN repo, the original name was inference_preprocess.py
def preprocess_raw_video(videoFilePath, dim=36):
    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT)) # get total frame size
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    print("Orignal Height", height)
    print("Original width", width)

    #########################################################################
    # Facial Detection & Cropping
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=30,
        minSize=(70, 70),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    x, y, w, h = faces[0]
    padding = 30

    if args.background_augmentation:
        bg_color = args.background_color
        background = np.ones(img.shape) * bg_color
    #########################################################################
    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC)) # current timestamp in millisecond
        frame = img

        # Background augmentation
        if args.background_augmentation:
            alpha_matte_path = os.path.join(args.alpha_matte_path, base_name + '/' + base_name + '_' +
                                            str(i).zfill(7) + '.png')
            matte = cv2.imread(alpha_matte_path) / 255
            frame = bg_aug_matte(frame, background, matte)

        vidLxL = cv2.resize(img_as_float(frame[y-padding:y+h+padding, x-padding:x+w+padding, :]), (dim, dim), interpolation=cv2.INTER_AREA)
        # vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)

        # Lighting augmentation
        if args.lighting_augmentation == "gamma":
            vidLxL = lighting_aug_gamma(vidLxL, args.lighting_params)
        if args.lighting_augmentation == 'alpha':
            vidLxL = lighting_aug_alpha(vidLxL, args.lighting_params[0], args.lighting_params[1])

        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1/255)] = 1/255
        Xsub[i, :, :, :] = vidLxL
        success, img = vidObj.read() # read the next one
        i = i + 1
    plt.imshow(Xsub[0])
    plt.title('Sample Preprocessed Frame')
    plt.show()

    #########################################################################
    # Normalized Frames in the motion branch
    normalized_len = min(len(t), totalFrames) - 1
    dXsub = np.zeros((normalized_len, dim, dim, 3), dtype=np.float32)
    for j in range(normalized_len - 1):
        dXsub[j, :, :, :] = (Xsub[j+1, :, :, :] - Xsub[j, :, :, :]) / (Xsub[j+1, :, :, :] + Xsub[j, :, :, :])

    dXsub = dXsub / np.std(dXsub)
    #########################################################################
    # Normalize raw frames in the appearance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub / np.std(Xsub)
    Xsub = Xsub[:min(len(t), totalFrames)-1, :, :, :]
    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis=3)
    return dXsub


def preprocess_raw_video_crop_motion_branch(videoFilePath, dim=36):
    #########################################################################
    # set up
    t = []
    i = 0
    vidObj = cv2.VideoCapture(videoFilePath)
    totalFrames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))  # get total frame size
    Xsub = np.zeros((totalFrames, dim, dim, 3), dtype=np.float32)
    dXsub = Xsub.copy()
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    success, img = vidObj.read()
    print("Orignal Height", height)
    print("Original width", width)

    #########################################################################
    # Facial Detection & Cropping
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascPath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(70, 70),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    x, y, w, h = faces[0]
    padding = 30

    bg_color = args.background_color
    background = np.ones(img.shape) * bg_color

    prev_frame = None     # Used to calculate motion branch
    #########################################################################
    # Crop each frame size into dim x dim
    while success:
        t.append(vidObj.get(cv2.CAP_PROP_POS_MSEC))  # current timestamp in millisecond
        frame = img

        # Background augmentation
        alpha_matte_path = os.path.join(args.alpha_matte_path, base_name + '/' + base_name + '_' +
                                        str(i).zfill(7) + '.png')
        matte = cv2.imread(alpha_matte_path) / 255
        frame = bg_aug_matte(frame, background, matte)

        vidLxL = cv2.resize(img_as_float(frame[y - padding:y + h + padding, x - padding:x + w + padding, :]), (dim, dim),
                            interpolation=cv2.INTER_AREA)
        # vidLxL = cv2.rotate(vidLxL, cv2.ROTATE_90_CLOCKWISE) # rotate 90 degree
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        Xsub[i, :, :, :] = vidLxL

        # Normalized Frames in the motion branch
        vidLxL = cv2.resize(img_as_float(img[y - padding:y + h + padding, x - padding:x + w + padding, :]), (dim, dim),
                            interpolation=cv2.INTER_AREA)
        vidLxL = cv2.cvtColor(vidLxL.astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        if i != 0:
            matte = cv2.resize(img_as_float(matte[y - padding:y + h + padding, x - padding:x + w + padding, :]), (dim, dim),
                               interpolation=cv2.INTER_AREA)

            # This line of code still don't work, need to find out why
            motion_frame = ((vidLxL - prev_frame) / (vidLxL + prev_frame)).astype(type(matte[0][0][0]))
            dXsub[i - 1, :, :, :] = cv2.bitwise_and(motion_frame, matte)

        prev_frame = vidLxL.copy()

        success, img = vidObj.read()  # read the next one
        i = i + 1
    plt.imshow(Xsub[0])
    plt.title('Sample Preprocessed Frame')
    plt.show()

    #########################################################################
    # Normalized Frames in the motion branch
    dXsub = dXsub[:min(len(t), totalFrames) - 1, :, :, :]
    dXsub = dXsub / np.std(dXsub)
    #########################################################################
    # Normalize raw frames in the appearance branch
    Xsub = Xsub - np.mean(Xsub)
    Xsub = Xsub / np.std(Xsub)
    Xsub = Xsub[:min(len(t), totalFrames) - 1, :, :, :]
    #########################################################################
    # Plot an example of data after preprocess
    dXsub = np.concatenate((dXsub, Xsub), axis=3)
    return dXsub

def vid2mat(data_dir):
    vid_path = os.path.join(data_dir, "vid.avi")
    label_path = os.path.join(data_dir, "ground_truth.txt")
    if args.crop_motion_branch:
        dXsub = preprocess_raw_video_crop_motion_branch(vid_path, 72)
    else:
        dXsub = preprocess_raw_video(vid_path, 72)
    dysub = None

    with open(label_path, 'r') as f:
        lines = [np.fromstring(line, dtype='float32', sep=' ') for line in f.readlines()]
        dysub = lines[0][:dXsub.shape[0]]
        dysub = np.reshape(dysub, (1, -1))

    drsub = np.zeros_like(dysub)

    matfiledata = {'dXsub': dXsub,
                   'dysub': dysub,
                   'drsub': drsub}

    return matfiledata


def generate_exp_name():
    result = "exp"
    if args.background_augmentation:
        result += '_bg_' + args.background_color
    if args.lighting_augmentation:
        result += "_" + args.lighting_augmentation + "_" + args.lighting_params
    if args.crop_motion_branch:
        result += "_motion_cropped"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', type=str, help='path of subject folders, each contain a vid.avi '
                                           'and groundtruth.txt')
    parser.add_argument('-o', '--output-dir', type=str, help='path of output folder, which contains all the .mat files')
    parser.add_argument('-bg-aug', '--background-augmentation', action='store_true', help="augment background")
    parser.add_argument('-bg-color', '--background-color', type=str, default='(0, 0, 0)', help="input rgb for background color default black. e.g. '(255, 255, 255)'")
    parser.add_argument('-m', '--alpha-matte-path', type=str, default=None, help="path of alpha matte folder, contains "
                                                                                 "P* folders, which has .png files")
    parser.add_argument('-l-aug', '--lighting-augmentation', type=str, default=None, help="gamma or alpha")
    parser.add_argument('-l-params', '--lighting-params', type=str, default=None, help="parameters for lighting augmentation, one number is required for gamma augmentation, and two numbers are required for alpha-beta transform, "
                                                                                       "e.g. 0.1  or '(80, 150)'")
    parser.add_argument('-c', '--crop-motion-branch', action='store_true', help="Only used for augmentation, produce "
                                                                                "the frames in the motion branch by:"
                                                                                "1. get difference between unaugmented frames"
                                                                                "2. crop the result"
                                                                                "Else, the frames in the motion branch"
                                                                                "will be generated from the augmented frames")
    args = parser.parse_args()

    exp_name = generate_exp_name()

    if args.background_augmentation:
        assert args.alpha_matte_path is not None, "Missing alpha matte path"
        assert args.background_color is not None, "Missing background color"
        args.background_color = np.fromstring(args.background_color.strip("()"), dtype='uint8', sep=',')
        assert args.background_color.shape[0] == 3, "Background color shape incorrect, getting{}".format(args.background_color.shape)

    if args.lighting_augmentation:
        assert args.lighting_augmentation.lower() in ["gamma", 'alpha'], "Lighint augmentation mode must be either gamma or alpha"
        assert args.lighting_params is not None, "Missing lighting augmentation parameters"
        args.lighting_params = np.fromstring(args.lighting_params.strip("()"), dtype='float32', sep=',')
        if args.lighting_augmentation == "alpha":
            assert len(args.lighting_params) == 2, "Alpha transform needs two parameters: (alpha, beta)"
        else:
            assert len(args.lighting_params) == 1, "Gamma transform needs one parameter: gamma"
            args.lighting_params = args.lighting_params[0]

    if args.crop_motion_branch:
        assert args.background_augmentation, "Crop Motion Branch is only used for data augmentation"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    exp_path = os.path.join(args.output_dir, exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    existing_files = os.listdir(exp_path)

    for subject_folder_name in os.listdir(args.input_dir):
        subject_folder_path = os.path.join(args.input_dir, subject_folder_name)
        base_name = "P" + subject_folder_name[7:]

        # Skip the existing .mat results
        file_name = base_name + '.mat'
        if file_name not in existing_files:
            print("Processing", file_name)
            matfiledata = vid2mat(subject_folder_path)
            output_path = os.path.join(exp_path, file_name)
            print("Saving {} under {}".format(file_name, exp_path) +
                  '\n \t dXsub: ', matfiledata["dXsub"].shape,  # (N, 72, 72, 6)
                  '\n \t dysub: ', matfiledata["dysub"].shape,  # (1, N)
                  '\n \t drsub: ', matfiledata["drsub"].shape)  # (1, N)

            hdf5storage.write(matfiledata, filename=output_path)
        else:
            print("Skipping {} because it already exists".format(file_name))
