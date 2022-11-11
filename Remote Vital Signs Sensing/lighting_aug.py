import h5py
import hdf5storage
import cv2
import os
import glob
import argparse
import numpy as np
import torch
import augmentations as aug

import matplotlib.pyplot as plt

# INPUT_PATH = "/home/risuka/cvresearch/data-blackback/black-background-fixed/UBFCChunks72x72_BackgroundAug_Black-001/UBFCChunks72x72_BackgroundAug_Black/P10C1.mat"

def examine_file(data_path, style:str):
    with h5py.File(data_path, 'r') as f:
        data = torch.tensor(np.array(f['dXsub']))
        print("Before Transformation: ", data.shape)
        data = data.permute(3, 2, 1, 0)[..., 3:6]
        print("After Transoformation: ", data.shape)
        data_flat = torch.flatten(data[0])
        data_max, data_min = max(data_flat), min(data_flat)
        print("Data Range: ", data_max, data_min)
        # print("Data sample", data[0])
        # for i in range(data.shape[0]):
        for i in range(1):
            img = data.detach().cpu().numpy()[i]
            # print(img.shape)
            img = np.float32(img) # Necessary for matplot to correctly plot the error
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Necessary for matplotlib, it uses a different format than cv2

            if style=='gamma':
                img_arr = transform_gamma(img)
            elif style=='alpha':
                img_arr = transform_alpha(img)
            else:
                img_arr = img

            plot_together(img_arr)
            cv2.waitKey(0) # 0 waits for keypress, positive integers specify how many ms to wait for

# TODO Add titles
def plot_together(img_arr):
    # TODO Make plotting automatic, not hardcoded cols and rows
    nrows = 3
    ncols = 7
    figsize = [8,10]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # alphas = [x for x in range(80, 200, 5)]
    gammas = [x/100 for x in range(5, 150, 5)]

    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        img = img_arr[i]
        axi.imshow(img)
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
        axi.set_title("G: " + str(gammas[i]))

    # plt.tight_layout(True)
    plt.show()

''' Takes an input image, applies several gamma transforms to it, returns output images as array '''
def transform_gamma(img):
    ''' Gamma range plotting '''
    gammas = [x/100 for x in range(5, 150, 5)]
    img_arr = [gammaCorrection(img, gamma) for gamma in gammas]
    return img_arr


''' Takes an input image, applies several alpha+beta transforms to it, returns output images as array '''
def transform_alpha(img):
    ''' Alpha range plotting '''
    alphas = [x for x in range(80, 200, 5)]
    img_arr = [cv2.convertScaleAbs(img, alpha=a, beta=150.0) for a in alphas]
    return img_arr

# Modified from the cv2 tutorial: https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
# Modified from other projects doing gamma transforms: https://www.programcreek.com/python/example/89460/cv2.LUT
def gammaCorrection(img_original, gamma):
    gamma_table = np.array([np.power(x / 255.0, gamma) * 255.0 for x in range(256)]).clip(0, 255).astype('uint8')
    img = img_original.clip(0, 255)
    dst = cv2.resize(img, (72, 72))
    cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)
    dst = dst.astype('uint8')
    print(gamma_table.shape)
    print(dst.shape)
    result = cv2.LUT(dst, gamma_table)
    return result #   np.clip(result, 0, 255) # No change from simply result

def extract(data_path):
    with h5py.File(data_path, 'r') as f:
        data = torch.tensor(np.array(f['dXsub']))
        data = data.permute(3, 2, 1, 0).detach().cpu().numpy()
        return data[..., 0:3], data[..., 3:6], np.array(f['dysub']), np.array(f['drsub'])

# Takes an image, outputs that image with its histogram equalized
def aug_equalize_histogram_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # TODO Test parameter effects
    return clahe.apply(img)

#
# def color_filter(frame, epsilon):
#     u_green, l_green = find_bg(frame, epsilon)
#     # I used the minimum value in the picture as black, could be a very small value as well
#     # val_black = np.min(frame)
#     val_black = -10
#     frame_black = np.zeros(frame.shape)
#     frame_black.fill(val_black)a
#
#     mask = cv2.inRange(frame, l_green, u_green)
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     f = frame - res
#     f = np.where(f == 0, frame_black, f)
#
#     dim = (frame.shape[1] * 4, frame.shape[0] * 4)
#
#     # frame = cv2.resize(frame, dim)
#     # f = cv2.resize(f, dim)
#     cv2.imshow('frame', frame)
#     cv2.imshow("mask", f)
#
# def test_colors(img):
#     cv2.imshow("test_colors", img)
#     epsilon = 1
#     while True:
#         k = chr(cv2.waitKey(0))
#         if k == 'w':
#             epsilon += 0.01
#         elif k == 's':
#             epsilon -= 0.01
#         elif k == 'x':
#             cv2.destroyAllWindows()
#             print("The epsilon is ", epsilon)
#             break
#         else:
#             continue
#         color_filter(img, epsilon)

# TODO Make the images larger; do the lighting on higher res images, rescale and use that as the input: benefits include easier to view

# if __name__ == '__main__':
#
#     # examine_file(INPUT_PATH)
#     motion, appearance, dysub, drsub = extract('C:\\Users\\Andre\\OneDrive\\Research\\CV for Telehealth\\Data\\black-background\\UBFCChunks72x72_BackgroundAug_Black\\P1\\P1C1.mat')
#     original = appearance[0]
#     # Gamma values: 0.05 ~ 1.5
#     # gammas = [x/100 for x in range(5, 150, 5)]
#     gammas = [x/100 for x in range(5, 150, 5)]
#     augmented_imgs = [gammaCorrection(original, gamma) for gamma in gammas]
#     cv2.imshow("Original", original)
#     for idx, augmented_img in enumerate(augmented_imgs):
#         cv2.imshow("Augmented", augmented_img)
#
#
#
#         cv2.waitKey(500)

# if __name__ == '__main__':
#
#     INPUT_PATH = 'C:\\Users\\Andre\\OneDrive\\Research\\CV for Telehealth\\Data\\black-background\\UBFCChunks72x72_BackgroundAug_Black\\P22\\P22C1.mat'
#     examine_file(INPUT_PATH)
#     motion, appearance, dysub, drsub = extract(INPUT_PATH)
#     original = appearance[0]
#     # Gamma values: 0.05 ~ 1.5
#     alphas = [x for x in range(80, 200, 5)]
#     augmented_imgs = [cv2.convertScaleAbs(original, alpha=a, beta=100.0) for a in alphas]
#     cv2.imshow("Original", original)
#     for idx, augmented_img in enumerate(augmented_imgs):
#         cv2.imshow("Augmented", augmented_img)
#         print("alpha: " + str(alphas[idx]))
#
#
#         cv2.waitKey(500)

if __name__ == '__main__':

    # # examine_file(INPUT_PATH)
    # motion, appearance, dysub, drsub = extract('C:\\Users\\Andre\\OneDrive\\Research\\CV for Telehealth\\Data\\black-background\\UBFCChunks72x72_BackgroundAug_Black\\P1\\P1C1.mat')
    # # print('Length of appearance: ' + str(len(appearance)))
    # original = appearance[0]
    # # Gamma values: 0.05 ~ 1.5
    # # gammas = [x/100 for x in range(5, 150, 5)]
    # gammas = [0.9]
    # augmented_imgs = [gammaCorrection(original, gamma) for gamma in gammas]
    # cv2.imshow("Original", original)
    #
    # for idx, augmented_img in enumerate(augmented_imgs):
    #     cv2.imshow("Augmented", augmented_img)
    #
    #     # Create a new appearance, process all images
    #     new_appearance = [gammaCorrection(app, gammas[idx]) for app in appearance]
    #
    #     output_path = 'C:\\Users\\Andre\\OneDrive\\Research\\CV for Telehealth\\Data\\aug-black-background\\P1\\' + 'P1C1' + 'G' + str(gammas[idx]) + '.mat'
    #     output_data = np.concatenate((motion, new_appearance), axis=-1)
    #     # print(output_data.shape)
    #     #
    #     # print('Saving file: ', output_path,
    #     #       '\n \t dXsub: ', output_data.shape,
    #     #       '\n \t dysub: ', dysub.shape,
    #     #       '\n \t drsub: ', drsub.shape)
    #
    #     matfiledata = {'dXsub': output_data,
    #                    'dysub': dysub,
    #                    'drsub': drsub}
    #
    #     hdf5storage.write(matfiledata, filename=output_path)
    #
    #     cv2.waitKey(0)



# if __name__ == '__main__':
#
#     # examine_file('./data/UBFCChunks72x72_BackgroundAug/P1_result/P1C1.mat')
#     # examine_file('C:\\Users\\Andre\\OneDrive\\Research\\CV for Telehealth\\Data\\UBFC-small\\UBFC72x72-20220209T041803Z-002\\UBFC72x72\\P8.mat')
#     examine_file('C:\\Users\\Andre\\OneDrive\\Research\\CV for Telehealth\\Data\\black-background\\UBFCChunks72x72_BackgroundAug_Black\\P1\\P1C1.mat')
#
#     # define cmd arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input-path', type=str, help='path of chunk folders')
#     parser.add_argument('--output-path', type=str, help='path of output folder')
#     args = parser.parse_args()
#
#     if not os.path.exists(args.input_path):
#         print('Cannot find input path: {0}'.format(args.input_path))
#         exit()
#     if not os.path.exists(args.output_path):
#         print('Cannot find output path: {0}'.format(args.output_path))
#         exit()
#
#     epsilon = 1.63
#
#     for file_name in os.listdir(args.input_path):
#         path = os.path.join(args.input_path, file_name)
#         output_path = os.path.join(args.output_path, file_name)
#
#         motion, appearance, dysub, drsub = extract(path)
#
#         # todo Andrey: appearance is the data I am working with, the other data is written back into the file as they are, I am augmenting appearance
#         # Examine file is for testing; once done, put that code into the main
#
#         # Take the pixel value at (w, h) = (0, 0)
#
#         # Use this line to find a good epsilon
#         # test_colors(cv2.cvtColor(data[0], cv2.COLOR_BGR2RGB))
#
#         # write data
#         # todo Andrey use this to write data to a file once I have good transformations
#         output_data = np.concatenate((motion, new_data), axis=-1)
#
#         print(output_data.shape)
#
#         print('Saving file: ', output_path,
#               '\n \t dXsub: ', output_data.shape,
#               '\n \t dysub: ', dysub.shape,
#               '\n \t drsub: ', drsub.shape)
#
#         matfiledata = {'dXsub': output_data,
#                        'dysub': dysub,
#                        'drsub': drsub}
#
#         hdf5storage.write(matfiledata, filename=output_path)