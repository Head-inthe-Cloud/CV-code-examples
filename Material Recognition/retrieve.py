import os
import pickle
import argparse
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np


import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights

from encode import encode, get_latent_features
from utils import euclidean, cosine_distance, build_montages


RANDOMSTATE = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--img_path", type=str, required=True,
                        help="path to the image you want to query")
    parser.add_argument("--data_dir", type=str, default='./data',
                        help="path to Dataset")
    parser.add_argument("--pickle_path", type=str, default=None,
                        help="path to the pkl file for the encoded dataset")
    parser.add_argument("--output_dir", type=str, default='./results',
                        help="path to save search results")
    parser.add_argument("--show_imgs", action='store_true', default=False,
                        help="visualize the search results")
    parser.add_argument("--num_query", type=int, default=10,
                        help="number of images to query")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Image size")

    return parser


def perform_search(queryFeatures, index, maxResults=64):
    results = []
    cosine = True
    for i in range(0, len(index["features"])):
        # compute the euclidean distance between our query features
        # and the features for the current image in our index, then
        # update our results list with a 2-tuple consisting of the
        # computed distance and the index of the image
        if cosine:
            d = cosine_distance(np.reshape(queryFeatures, -1), np.reshape(index["features"][i], -1))
        else:
            d = euclidean(queryFeatures, index["features"][i])

        results.append((d, i))

    # sort the results and grab the top ones
    results = sorted(results)[:maxResults]
    # return the list of results
    return results


def retrieve(img, data_dir, output_dir, pickle_path=None, show_imgs=False, num_query=10, img_size=512):
    if pickle_path is None:
        index_dict = encode(data_dir, output_dir)
    else:
        with open(pickle_path, 'rb') as f:
            index_dict = pickle.load(f)

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load Model in Evaluation phase
    # model = resnet50(weights=ResNet50_Weights.DEFAULT).cuda()
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights).cuda()
    model.eval()
    query_features = get_latent_features(model, img, transformations)

    results = perform_search(query_features, index_dict, num_query)

    df = pd.DataFrame()
    df['image'] = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
                   os.path.isfile(os.path.join(data_dir, f))]
    image_paths = df.image.values

    outputs = []
    for score, index in results:
        print(score)
        path = image_paths[index]
        outputs.append([score, path])

    if show_imgs:
        cv2.imshow("original", cv2.cvtColor(cv2.resize(img, (img_size, img_size)), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        for score, path in outputs:
            cv2.imshow('Result', cv2.imread(path))
            cv2.waitKey(0)
    else:
        print('Found {} results'.format(num_query))

    return outputs

if __name__ == '__main__':
    opts = get_argparser().parse_args()
    retrieve(cv2.cvtColor(cv2.imread(opts.img_path), cv2.COLOR_BGR2RGB),
             opts.data_dir,
             opts.output_dir,
             opts.pickle_path,
             opts.show_imgs,
             opts.num_query,
             opts.img_size)


    
