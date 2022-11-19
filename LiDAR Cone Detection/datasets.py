import csv

# If point_like is true, return [[angle, range, intensity] x N]
# else return [[angles], [range], [intensity]]
def extract_from_csv(file_path, point_like=True):
    f = open(file_path, 'r', newline='')
    reader = csv.reader(f)  
    dataset = []
    temp = 0
    if point_like:
            frame = []
            for i, row in enumerate(reader):
                    if i == 0:
                            continue
                    if i == 1:
                            temp = float(row[0])
                    if temp != float(row[0]):
                            dataset.append(frame)
                            temp = float(row[0])
                            frame = []
                    frame.append([float(row[1]), float(row[2]), float(row[3])])
            dataset.append(frame)
    
    else: 
            angle = []
            ran = []
            intensity = []
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