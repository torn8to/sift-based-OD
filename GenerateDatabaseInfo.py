import cv2
from matplotlib import pyplot as plt
import os
import pickle
from SiftHelperFunctions import *


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def scan_folder(folder_path):
    _file_list = []
    _data = []
    listing = os.listdir(folder_path)
    # listing.remove('.git')
    for file in listing:
        if os.path.isdir(folder_path + file):
            temp_data, temp_files = scan_folder(folder_path+file+"/")
            _data.extend(temp_data)
            _file_list.extend(temp_files)
        else:
            _file_list.append(file)
            img = cv2.imread(folder_path + file)

            # image comparison
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # need to do some more processing here

            # find the keypoints and descriptors with SIFT
            kp, des = sift.detectAndCompute(gray_img, None)
            img_size = (len(gray_img[0]), len(gray_img))

            # centroid = get_centroid(kp)
            centroid = (img_size[0]/2, img_size[1]/2)

            temp_kp = make_temp_kp(kp)
            datum = [temp_kp, des, img_size, centroid, path1 + file]
            _data.append(datum)
    return _data, _file_list


def get_files(folder_path):
    listing = os.listdir(folder_path)
    files = []
    # listing.remove('.git')
    for file in listing:
        if os.path.isdir(folder_path + file):
            files.extend(get_files(folder_path + file + "/"))
        else:
            files.append(file)
    return files


if __name__ == "__main__":
    sift = cv2.SIFT_create()
    path1 = '../Data_Set/Query dataset/train dataset/'
    data, file_list = scan_folder(path1)
    save_loc = '../Data_Set/training_data.pkl'
    save_object(data, save_loc)
    print("Data file saved: ", save_loc)
