import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str,
                    help='data directory')
parser.add_argument('-m', '--mode', type=str,
                    help='organize mode, c for combine and s for split')
args = parser.parse_args()

# Put files in the data_dir into separate folders
def combine():
    data_dir = args.dir
    file_names = os.listdir(data_dir)
    dir_names = []
    for file_name in file_names:
        dir_name = file_name.split('C')[0]
        dir_path = os.path.join(data_dir, dir_name)
        if dir_name not in dir_names:
            os.mkdir(dir_path)
            dir_names.append(dir_name)
        file_path = os.path.join(data_dir, file_name)
        new_file_path = os.path.join(dir_path, file_name)
        os.rename(file_path, new_file_path)

# Take files out from separate folders
def split():
    data_dir = args.dir
    dir_names = os.listdir(data_dir)
    for dir_name in dir_names:
        dir_path = os.path.join(data_dir, dir_name)
        file_names = os.listdir(dir_path)
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            new_file_path = os.path.join(data_dir, file_name)
            os.rename(file_path, new_file_path)
        os.rmdir(dir_path)

if __name__ == '__main__':
    if args.mode == 'c':
        combine()
    if args.mode == 's':
        split()