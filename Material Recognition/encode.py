import os
import argparse

import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'mobilenet'


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_dir", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--output_dir", type=str, default='./encoded',
                        help="path to save encoded data")

    return parser


def get_latent_features(model, images, transformations):
    # latent_features = np.zeros((4738, 256, 16, 16))
    # latent_features = np.zeros((4738,8,42,42))
    if type(images) is str:
        img = Image.open(images)
        if images.endswith('.png'):
            img = img.convert('RGB')
        tensor = transformations(img).to(device)
        return model(tensor.unsqueeze(0)).cpu().detach().numpy()

    if type(images) is np.ndarray and len(images.shape) == 3:
        tensor = transformations(images).to(device)
        return model(tensor.unsqueeze(0)).cpu().detach().numpy()

    latent_features = []

    for i, image in enumerate(tqdm(images)):
        img = Image.open(image)
        if image.endswith('.png'):
            img = img.convert('RGB')
        tensor = transformations(img).to(device)
        latent_features.append(model(tensor.unsqueeze(0)).cpu().detach().numpy())

    del tensor
    # gc.collect()
    return latent_features


def encode(data_dir, output_dir, save_pkl=False):
    dataset_path = data_dir
    df = pd.DataFrame()

    df['image'] = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if
                   os.path.isfile(os.path.join(dataset_path, f))]
    print(df.head())

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load Model in Evaluation phase
    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_name == 'mobilenet':
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights).cuda()
    model.eval()

    print(model.to(device))

    images = df.image.values
    latent_features = get_latent_features(model, images, transformations)

    indexes = list(range(0, len(images)))
    feature_dict = dict(zip(indexes, latent_features))
    index_dict = {'indexes': indexes, 'features': latent_features}

    if save_pkl:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        pkl_path = os.path.join(output_dir, 'features.pkl')
        with open(pkl_path, "wb") as f:
            f.write(pickle.dumps(index_dict))

    return index_dict

if __name__ == "__main__":
    opts = get_argparser().parse_args()
    encode(opts.data_dir, opts.output_dir, True)
