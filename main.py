# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
# from numpy.lib.arraysetops import unique
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
import math
from SiftHelperFunctions import *
from HoughTransform import *
from PostProcessing import *
from VisualHelperFunctions import *


# Initiate SIFT detector
sift = cv2.SIFT_create()


image_query = cv2.imread('/home/prajwal/Desktop/cv_group_project/Data_Set/Test_DataSet/other/Test_image_09.jpg')  # Query Image
rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
kp_query, des_query = sift.detectAndCompute(gray_query, None)

img_size_list = []
img_centroid_list = []
path = '/home/prajwal/Desktop/cv_group_project/Data_Set/Train_DataSet/DatabaseInfo/standing/training_data.pkl'
with open(path, 'rb') as inp:
    data = pickle.load(inp)

    temp_kp = data[0][0]
    img_size_list = [data[0][2]] * len(data[0][0])
    img_centroid_list = [data[0][3]] * len(data[0][0])
    temp_des = data[0][1]
    for datum in data[1:]:
        temp_kp.extend(datum[0])
        img_size_list.extend([datum[2]] * len(datum[0]))  # maintaining list of img_size for each keypoint
        img_centroid_list.extend([datum[3]] * len(datum[0]))  # maintain centroid for each keypoint
        temp_des = np.append(temp_des, datum[1],axis=0)

# Organize model key points and descriptors into single vector/matrix
kp = make_kp(temp_kp)
des = temp_des

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_query, des, k=2)  # query ,database,nearest neighbors

# Apply ratio test
good_matches = []
matching_keypoints = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
        matching_keypoints.append((kp[m.trainIdx], kp_query[m.queryIdx],  img_size_list[m.trainIdx], img_centroid_list[m.trainIdx]))

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

# Perform Hough Transform
hough_dict = perform_hough_transform(matching_keypoints, image_query)

valid_bins = []
keypoint_pairs = []
keys = hough_dict.keys()

# Keep all centroids with votes greater than 5 and discard the rest

for i, key in enumerate(keys):
    # print(hough_dict[key].votes)
    if hough_dict[key].votes >= 5:
        keypoint_pairs.extend(hough_dict[key].keypoint_pairs)
        if hough_dict[key] not in valid_bins:
            valid_bins.append(hough_dict[key])

##Call the Affine Transformation function here.
##Valid bins is a list of PoseBin object with each object having votes greater than 5

# Calculates average pose for each cluster based on position and orientation
final_pose = post_process(valid_bins)

fig, ax = plot_rect(rgb_query, final_pose, keypoint_pairs)
# fig, ax = plot_keypoint(rgb_query, keypoint_pairs)
plt.show()
print("done")