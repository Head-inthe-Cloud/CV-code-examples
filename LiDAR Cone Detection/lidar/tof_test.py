import os
import ydlidar
import time
import sys
import csv
# from plot import plot

DATA_NAME = 'data/12202021/lidar_data_noisy_1m_2.csv'
if __name__ == "__main__":
    ydlidar.os_init();
    ports = ydlidar.lidarPortList();
    port = "/dev/ydlidar";
    for key, value in ports.items():
        port = value;
    laser = ydlidar.CYdLidar();
    laser.setlidaropt(ydlidar.LidarPropSerialPort, port);
    laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 128000); # Number of symbols transmitted per second, one symbol could be one or more bits
    laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TOF);
    laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL);
    laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0); # 10 Hz scan frequency, 10 circles per seconds
    laser.setlidaropt(ydlidar.LidarPropSampleRate, 5);
    laser.setlidaropt(ydlidar.LidarPropSingleChannel, False);

    ret = laser.initialize();
    if ret:
        ret = laser.turnOn();
        scan = ydlidar.LaserScan()
        # This opens the data file, or creates a new one if there isn't one
        f = open(DATA_NAME, 'w', newline='')
        writer = csv.writer(f)
        # Headers for data
        writer.writerow(['time_stamp', 'angle', 'range', 'intensity'])
        '''
        while ret and ydlidar.os_isOk():
            r = laser.doProcessSimple(scan);
            if r:
                angle = []
                ran = []
                intensity = []
                for point in scan.points:
                    angle.append(point.angle);
                    ran.append(point.range);
                    intensity.append(point.intensity);
                    # This line writes a row in .csv file, which represents a dot detected by the lidar
                    writer.writerow([scan.stamp, point.angle, point.range, point.intensity])
                    print("Angle: ", point.angle, " Range: ", point.range, " Intensity: ", point.intensity)
                print("Scan received[",scan.stamp,"]:",scan.points.size(),"ranges is [",1.0/scan.config.scan_time,"]Hz");
            else:
                print("Failed to get Lidar Data.")
            time.sleep(0.05);
        '''
        for i in range(100):
            r = laser.doProcessSimple(scan);
            if r:
                angle = []
                ran = []
                intensity = []
                for point in scan.points:
                    angle.append(point.angle);
                    ran.append(point.range);
                    intensity.append(point.intensity);
                    # This line writes a row in .csv file, which represents a dot detected by the lidar
                    writer.writerow([scan.stamp, point.angle, point.range, point.intensity])
                    print("Angle: ", point.angle, " Range: ", point.range, " Intensity: ", point.intensity)
                print("Scan received[", scan.stamp, "]:", scan.points.size(), "ranges is [", 1.0 / scan.config.scan_time,
                      "]Hz");
            else:
                print("Failed to get Lidar Data.")
            time.sleep(0.05)
        laser.turnOff();
        # Don't forget to close the file when you are done
        f.close()
    laser.disconnecting();
    #
    #
    #
    # plot()
