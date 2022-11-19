import matplotlib.pyplot as plt
import csv

RMAX = 32.0
NUM_SHOW = 1   # how many data in the file do you wish to see?

# Specify the data file path
data_file = 'data/12052021/lidar_data_2m.csv'


def read_file(file_name):
        f = open(file_name, 'r', newline='')
        reader = csv.reader(f)
        dataset = []
        angle = []
        ran = []
        intensity = []
        temp = 0
        for i, row in enumerate(reader):
                if i == 0:
                        continue
                if i == 1:
                        temp = float(row[0])
                if temp != float(row[0]):
                        dataset.append([angle, ran, intensity])
                        angle = []
                        ran = []
                        intensity = []
                        temp = float(row[0])
                angle.append(float(row[1]))
                ran.append(float(row[2]))
                intensity.append(float(row[3]))
        dataset.append([angle, ran, intensity])

        return dataset

def write_file(dataset, file_name):
        f = open(file_name, 'w', newline='')
        fieldnames = ["index", "angle", "range", "intensity"]
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for index, data in enumerate(dataset):
                for i in range(len(data[0])):
                        writer.writerow([index, data[0][i], data[1][i], data[2][i]])

def plot(dataset, num_show = NUM_SHOW):

        for i, data in enumerate(dataset):
                if i >= num_show:
                        break
                # fig = plt.figure()
                # fig.canvas.set_window_title('YDLidar LIDAR Monitor')
                lidar_polar = plt.subplot(polar=True)
                lidar_polar.autoscale_view(True,True,True)
                lidar_polar.set_rmax(RMAX)
                lidar_polar.grid(True)

                lidar_polar.scatter(data[0], data[1], c=data[2], cmap='hsv', alpha=0.95)
                # lidar_polar.scatter(angle, ran, cmap='hsv', alpha=0.95)



                # lidar_polar.clear()
                # lidar_polar.scatter(angle, ran, alpha=0.95)
                plt.show()


if __name__ == "__main__":
        plot(read_file(data_file))
