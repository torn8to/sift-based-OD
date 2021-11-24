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


# Initiate SIFT detector
sift = cv2.SIFT_create()


image_query = cv2.imread('/home/prajwal/Desktop/cv_group_project/Sift-Implementation/Test_images/Test_image9.jpg')  # Query Image
rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
kp_query, des_query = sift.detectAndCompute(gray_query, None)

img_size_list = []
img_centroid_list = []
with open('training_data.pkl', 'rb') as inp:
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
# model_kp = []
# shape_model_kp = []  ##stores the information of shape image from which the keypoint was generated
# queryImage_kp = []
# centroid_model = []  ##stores the centroid information corresponding to image keypoint
matching_keypoints = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
        # model_kp.append(kp[m.trainIdx])
        # shape_model_kp.append(shape[m.trainIdx])
        # centroid_model.append(centroid[m.trainIdx])
        # queryImage_kp.append(kp_query[m.queryIdx])
        matching_keypoints.append((kp[m.trainIdx], kp_query[m.queryIdx],  img_size_list[m.trainIdx], img_centroid_list[m.trainIdx]))

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

hough_dict = perform_hough_transform(matching_keypoints, image_query)

best_pose = []
keypoint_pairs = []
keys = hough_dict.keys()

for i, key in enumerate(keys):
    if hough_dict[key][0] >= 5:
        keypoint_pairs.append(hough_dict[key][2])
        element = hough_dict[key][3][0], hough_dict[key][3][1], hough_dict[key][3][2], hough_dict[key][1]
        if element not in best_pose:
            best_pose.append(element)
        
## Group all nearby best poses based on position only
pose_cluster = group_position(best_pose)
##Group each pose_cluster based on orientation
orientation_cluster = group_orientation(pose_cluster)
## For each pose cluster, check the maximum orientation cluster and append in to a list
final_orientation_list = find_max_orientation(orientation_cluster)
## Final pose is the average of pose_cluster position, maximum of orienation cluster for each position,
#  average scale for each pose_cluster and minimum image area for each pose_cluster
final_pose = get_final_pose(pose_cluster, final_orientation_list)

fig, ax = plt.subplots()
img = cv2.drawKeypoints(rgb_query, [x[1] for x in keypoint_pairs], None, flags=4)
plt.imshow(img)

for pose in final_pose:
    print("Object pose are:")
    print(pose)
    x_prime = - pose[3][1] *(pose[2]) / 2
    y_prime = - pose[3][0] * (pose[2]) / 2
    theta = pose[1]
    x_dash = math.cos(theta) * x_prime - math.sin(theta) * y_prime
    y_dash = math.sin(theta) * x_prime + math.cos(theta) * y_prime
    x = pose[0][0] + x_dash
    y = pose[0][1] + y_dash
    rect_left_corner = x, y
    rect = patches.Rectangle(rect_left_corner,
                        pose[3][1] * pose[2], pose[3][0] * pose[2], math.degrees(pose[1]),
                        linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
plt.show()
print("done")