import os
import posixpath
import h5py
import hdf5storage
import cv2
import numpy as np
import torch
import argparse


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
        data = data.permute(3, 2, 1, 0)
        label = torch.tensor(np.array(f['dysub']))
        _ = torch.tensor(np.array(f['drsub']))
        return data, label, _

if __name__ == "__main__":
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of chunk folders')
    parser.add_argument('--output-path', type=str, help='path of output folder')
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()

    # for test purpose
    # BASE_PATH = './data/UBFCChunks72x72_BackgroundAug/P10'
    # num_files = len(os.listdir(BASE_PATH))

    file_names = os.listdir(args.input_path)
    num_files = len(file_names)

    assert os.path.isdir(args.output_path)

    for file_name in file_names:
        # print("examining ", path)
        # examine_file(path)

        path = posixpath.join(args.input_path, file_name)

        data, label, _ = extract(path)

        total_frames = data.shape[0]
        num_chunks = int(np.floor(total_frames / 180))

        for idx in range(num_chunks):
            output_name = file_name[:-4] + 'C' + str(idx + 1) + '.mat'
            output_data = data[idx * 180:(idx + 1) * 180, ...]
            output_label = label[idx * 180:(idx + 1) * 180]
            output_drsub = _[idx * 180:(idx + 1) * 180]
            output_path = posixpath.abspath(posixpath.join(args.output_path, output_name))

            print('Saving file: ', output_path,
                  '\n \t dXsub: ', output_data.detach().cpu().numpy().shape,
                  '\n \t dysub: ', output_label.detach().cpu().numpy().shape,
                  '\n \t drsub: ', output_drsub.shape)

            matfiledata = {'dXsub': output_data.detach().cpu().numpy(),
                           'dysub': output_label.detach().cpu().numpy(),
                           'drsub': output_drsub.detach().cpu().numpy()}

            hdf5storage.write(matfiledata, filename=output_path)



