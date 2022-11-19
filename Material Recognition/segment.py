import cv2
import pymeanshift as pms
import numpy as np


# This is the Pascal 2012 colormap, used to turn number labels to rgb values, 1D images to rgb images
# Usage new_image = cmap[image]
def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def segment(img, img_size, spatial_radius=6, range_radius=15, min_density=70, show_img=False):
    cmap = color_map()
    original_image = cv2.resize(img, (img_size, img_size))

    (segmented_image, labels_image, number_regions) = pms.segment(original_image, spatial_radius=spatial_radius,
                                                                  range_radius=range_radius, min_density=min_density)

    if show_img:
        cv2.imshow('Original', original_image)
        cv2.imshow('Segmented', segmented_image)
        cv2.imshow('Labeled', cmap[labels_image])
        print(labels_image.shape)
        print(number_regions)
        cv2.waitKey(0)
    return labels_image


if __name__ == "__main__":
    path = "/home/patrick/CVProjects/Curie/test_data/vans_48/images/Vans-Sk8-Hi-Slim-Studded-Star-Shoes-_247530-front_jpg.rf.ad5de3cc22850556e58463043debde73.jpg"
    size = 512
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segment(image, size)


