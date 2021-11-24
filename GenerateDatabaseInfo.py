import cv2
from matplotlib import pyplot as plt
import os
import pickle
from SiftHelperFunctions import *


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)



data = []

sift = cv2.SIFT_create()
path1 = '/home/prajwal/Desktop/cv_group_project/Sift-Implementation/My_Data_Set/'
listing = os.listdir(path1)
# listing.remove('.git')
for file in listing:
    img = cv2.imread(path1 + file)
    img_size = img.shape
    # image comparison
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # need to do some more processing here
    
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(gray_img, None)
    centroid = get_centroid(kp)
    temp_kp = make_temp_kp(kp) ##keeping track of image height and width for each keypoint
    datum = [temp_kp, des, img_size, centroid, path1 + file]
    data.append(datum)
save_object(data, 'training_data.pkl')
