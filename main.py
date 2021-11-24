import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
from cv2 import sort
from SiftHelperFunctions import *
from HoughTransform import *
from PoseBin import *

# Initiate SIFT detector
sift = cv2.SIFT_create()

image_query = cv2.imread('../Data_Set/IMG_rotated_prajwal.jpg')  # Query Image
rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
kp_query, des_query = sift.detectAndCompute(gray_query, None)

img_size_list = []
img_centroid_list = []
with open('training_data.pkl', 'rb') as inp:
    data = pickle.load(inp)  # Open training data and load it

    temp_kp = data[0][0]  # temporary kp, grab the first element in the data set
    img_size_list = [data[0][2]] * len(data[0][0])
    img_centroid_list = [data[0][3]] * len(data[0][0])
    temp_des = data[0][1]  # the descriptor vector, grab the first element in the data set
    for datum in data[1:]:  # for the remaining elements append them to the previous two lists
        temp_kp.extend(datum[0])  # have to use extend here, because we don't want cascading lists
        img_size_list.extend([datum[2]] * len(datum[0]))  # maintaining list of img_size for each keypoint
        img_centroid_list.extend([datum[3]] * len(datum[0]))  # maintain centroid for each keypoint
        temp_des = np.append(temp_des, datum[1], axis=0)  # for numpy vectors we append

# Organize model key points and descriptors into single vector/matrix
kp = make_kp(temp_kp)
des = temp_des

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_query, des, k=2)  # query ,database,nearest neighbors

# Apply ratio test
good_matches = []
matching_keypoints = []  # Tuples of (kpM, kpQ)
for m, n in matches:
    if m.distance < 0.75 * n.distance or m.distance < 1.0:
        good_matches.append([m])
        # Store the matching keypoints in a tuple in a list
        matching_keypoints.append((kp[m.trainIdx], kp_query[m.queryIdx],
                                   img_size_list[m.trainIdx], img_centroid_list[m.trainIdx]))

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

# cv2.drawMatchesKnn expects list of lists as matches.
# img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=2)

pose_bins = perform_hough_transform(matching_keypoints)
# TODO use pose bins not dictionary
des_img_size = (0, 0)
keypoint_pairs = []
valid_bins = []  # A list of PoseBin objects
max_vote = 3
for key in pose_bins:
    if pose_bins.get(key).votes > 3:
        valid_bins.append(pose_bins.get(key))
    if pose_bins.get(key).votes > max_vote:
        print(pose_bins.get(key).votes, " votes for pose ", pose_bins.get(key))
        max_pose = key
        max_vote = pose_bins.get(key).votes
        des_img_size = pose_bins.get(key).img_size
        keypoint_pairs = pose_bins.get(key).keypoint_pairs
print("Most Voted Pose: ", max_pose)
print("Box Size: ", des_img_size)

# VISUALIZATION ###############################################################
fig, ax = plt.subplots()
img = cv2.drawKeypoints(gray_query, [x[1] for x in keypoint_pairs], None, None, flags=4)
plt.imshow(img)
# add box to image
IMG_WIDTH = des_img_size[0]
IMG_HEIGHT = des_img_size[1]
x_shift = -IMG_WIDTH * max_pose[3] / 2
y_shift = -IMG_HEIGHT * max_pose[3] / 2

# Determining the top left corner of the triangle with rotation
rect_left_corner = (max_pose[0] + np.cos(np.deg2rad(max_pose[2])) * x_shift - np.sin(np.deg2rad(max_pose[2])) * y_shift,
                    max_pose[1] + np.sin(np.deg2rad(max_pose[2])) * x_shift + np.cos(np.deg2rad(max_pose[2])) * y_shift)

rect = patches.Rectangle(rect_left_corner,
                         IMG_WIDTH * max_pose[3], IMG_HEIGHT * max_pose[3], max_pose[2],
                         linewidth=4, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()
print("done")
