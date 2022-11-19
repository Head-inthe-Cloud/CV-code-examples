import matplotlib.colors as mcolors
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import numpy as np

from datasets import extract_from_csv
from utils import toar

RMAX = 32.0
NUM_SHOW = 2  # how many data in the file do you wish to see?

# Specify the data file path
DATA_FILE = 'data/12202021/lidar_data_3m.csv'


# Plot a frame with points in the form of [Angle, Rnage] or [Angle, Range, Intensity]
def plot_ar(frame1, frame2=None, color=None):

    frame1 = np.transpose(frame1)
    if frame2 is not None:
        frame2 = np.transpose(frame2)

    lidar_polar = plt.subplot(polar=True)
    lidar_polar.autoscale_view(True, True, True)
    lidar_polar.set_rmax(RMAX)
    lidar_polar.grid(True)

    if color:
        lidar_polar.scatter(frame2[0], frame2[1], c=color, cmap='hsv', alpha=0.95)
    else:
        lidar_polar.scatter(frame1[0], frame1[1], c='r', alpha=0.95)
        if frame2 is not None:
            lidar_polar.scatter(frame2[0], frame2[1], c='g', alpha=0.95)

    plt.show()


def plot_xy(frame1, frame2=None, print_circle=False):
    fig, ax = plt.subplots()
    if print_circle:
        ax.plot([3], [0], 'b.')
    frame1 = np.transpose(frame1)
    ax.plot(frame1[0], frame1[1], 'r.')

    if frame2 is not None:
        if np.shape(frame2)[1] <= 3:
            frame2 = np.transpose(frame2)
        ax.plot(frame2[0], frame2[1], 'g.')
    if print_circle:
        ax.add_patch(plt.Circle((3, 0), 0.23/2, color='b', fill=False))

    plt.show()


def plot_eval(samples_xy, predictions):
    if np.shape(samples_xy)[1] <= 3:
        samples = np.transpose(toar(samples_xy))
        samples_xy = np.transpose(samples_xy)

    lidar_polar = plt.subplot(polar=True)
    lidar_polar.autoscale_view(True, True, True)
    lidar_polar.set_rmax(RMAX)
    lidar_polar.grid(True)

    lidar_polar.scatter(samples[0], samples[1], c=predictions, cmap='hsv', alpha=0.95)
    plt.title("Sampled points & Pr(Location | data)")

    plt.show()

    normalize = mcolors.Normalize(vmin=predictions.min(), vmax=predictions.max())
    colormap = cm.jet
    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(predictions)
    plt.colorbar(scalarmappaple)

    plt.scatter(samples_xy[0], samples_xy[1], c=colormap(normalize(predictions)))
    plt.title("Sampled points & Pr(Location | data)")
    plt.show()

if __name__ == "__main__":
    dataset = extract_from_csv(DATA_FILE, False)
    for i in range(NUM_SHOW):
        data = dataset[i]
        print(np.shape(data))
        plot_ar(data)
