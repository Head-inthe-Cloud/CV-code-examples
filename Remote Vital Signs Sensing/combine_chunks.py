import os
import h5py
import hdf5storage
import cv2
import numpy as np
import torch
import argparse
import re


def examine_file(data_path):
    with h5py.File(data_path, 'r') as f:
        data = torch.tensor(np.array(f['dXsub']))
        print("Before Transformation: ", data.shape)
        data = data.permute(3, 2, 1, 0)
        print("After Transoformation: ", data.shape)
        print("Data Sample: First -- ", data[0][0][0][0], "     Last -- ", data[-1][-1][-1][-1])
        label = torch.tensor(np.array(f['dysub']))
        print("Label Shape", label.shape)
        print("Label Sample: First -- ", label[0], "     Last -- ", label[-1])


def show_file(data_path):
    with h5py.File(data_path, 'r') as f:
        data = torch.tensor(np.array(f['dXsub'])).permute(3, 2, 1, 0)
        print(data.shape)

        for i in range(data.shape[0]):
            if i % 2 == 0:
                img = data[i, :, :, 3:6]
                img = img.detach().cpu().numpy()
                cv2.imshow('img', img)
                cv2.waitKey(0)


def extract(file_path):
    with h5py.File(file_path, 'r') as f:
        data = torch.tensor(np.array(f['dXsub']))
        data = data.permute(3, 2, 1, 0).detach().cpu().numpy()
        label = np.array(f['dysub'])
        _ = np.array(f['drsub'])
        return data, label, _

if __name__ == "__main__":
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

    assert os.path.isdir(args.output_path)

    # for test purpose
    # BASE_PATH = './data/UBFCChunks72x72_BackgroundAug/P10'
    # num_files = len(os.listdir(BASE_PATH))

    dir_names = os.listdir(args.input_path)

    for dir_name in dir_names:
        input_dir_path = os.path.join(args.input_path, dir_name)
        output_dir_path = args.output_path

        file_names = os.listdir(input_dir_path)
        file_names.sort(key=lambda f: int(re.sub('\D', '', f)))
        output_file_name = file_names[0].split("C")[0]   # we will add the .mat later
        output_path = os.path.join(output_dir_path, output_file_name + ".mat")

        total_frames = 0
        output_data = None
        output_label = None
        output_drsub = None

        for file_name in file_names:
            print('Extracting data from {}'.format(file_name))
            assert output_file_name in file_name    # Check if all chunks belong to one subject

            path = os.path.join(input_dir_path, file_name)
            data, label, _ = extract(path)

            total_frames += data.shape[0]
            if output_data is None:
                output_data = data
                output_label = label
                output_drsub = _
            else:
                output_data = np.concatenate((output_data, data))
                output_label = np.concatenate((output_label, label), axis=1)
                output_drsub = np.concatenate((output_drsub, _), axis=1)

        print('Saving file: ', output_path,
              '\n \t dXsub: ', output_data.shape,
              '\n \t dysub: ', output_label.shape,
              '\n \t drsub: ', output_drsub.shape)

        matfiledata = {'dXsub': output_data,
                       'dysub': output_label,
                       'drsub': output_drsub}

        hdf5storage.write(matfiledata, filename=output_path)



