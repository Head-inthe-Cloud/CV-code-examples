import math
import os

GPS_E = 47.658727
GPS_N = -122.314373
locationStd = 10
EARTH_RADIUS = 6371e3


class ConeClass:
    def __init__(self, ID, LM_GPS_E=None, LM_GPS_N=None, diameter=None):
        self.ID = ID
        self.LM_GPS_E = LM_GPS_E
        self.LM_GPS_N = LM_GPS_N
        self.diameter = diameter

class Vehicle:
    def __init__(self):
        self.closestCone = None


def getCone():
    cones = []      #Returns a list of all cones objects
    thisCone = None
    for line in open(PATH + '/' + 'ConeData.txt'):
        newConeFlag = 'Cone'
        if newConeFlag in line:
            ID = line.strip().split(':')[1]
            if not thisCone:
                thisCone = ConeClass(ID)
                continue
            cones.append(thisCone)
            thisCone = ConeClass(ID)
        if 'LM_GPS_E' in line:
            thisCone.LM_GPS_E = line.strip().split(':')[1]
        if 'LM_GPS_N' in line:
            thisCone.LM_GPS_N = line.strip().split(':')[1]
        if 'diameter' in line:
            thisCone.diameter = line.strip().split(':')[1]
    cones.append(thisCone)
    return cones

def printCone(cones):
    for cone in cones:
        print('Cone.ID:', cone.ID, 'LM_GPS_E:', cone.LM_GPS_E, 'LM_GPS_N:', cone.LM_GPS_E, 'diameter:', cone.diameter)

def FindClosestCone(cones, gps_e, gps_n):
    dist_to_closest_cone = math.inf

    for cone in cones:
        cone_Vcood_X = EARTH_RADIUS* (cone.LM_GPS_E - gps_e) * math.pi / 180 * math.cos(cone.GPS_N * math.pi /180)
        cone_Vcood_Y = EARTH_RADIUS* (cone.LM_GPS_N - gps_n) * math.pi / 180
        dist_sqr_to_cone = cone_Vcood_X**2 + cone_Vcood_Y**2

        if dist_sqr_to_cone < dist_to_closest_cone:
            dist_to_closest_cone = dist_sqr_to_cone
            closestCone = cone
    return closestCone

PATH = dir_path = os.path.dirname(os.path.realpath(__file__))

cones = getCone()
printCone(cones)

mAV = Vehicle()

cond = True

while cond:
    gps_e = GPS_E
    gps_n = GPS_N

    mAV.closestCone = FindClosestCone(cones, gps_e, gps_n)

    cond = False


    

    


        
