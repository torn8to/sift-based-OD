import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
from cv2 import sort
from SiftHelperFunctions import *
from VisualHelperFunctions import *
from HoughTransform import *
from PoseBin import *


#def main():
# Initiate SIFT detector
sift = cv2.SIFT_create()


image_query = cv2.imread('../Data_Set/Test dataset/clutter/IMG_3485.JPG')  # Query Image

rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
kp_query, des_query = sift.detectAndCompute(gray_query, None)
image_query_size = (len(gray_query[0]), len(gray_query))

img_size_list = []
img_centroid_list = []
with open('../Data_Set-multiple-train/training_data.pkl', 'rb') as inp:
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

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

# Apply ratio test
good_matches = []
matching_keypoints = []  # Tuples of (kpM, kpQ)
for m, n in matches:
    if m.distance < 0.75 * n.distance or m.distance < 1.0:
        good_matches.append([m])
        # Store the matching keypoints in a tuple in a list
        matching_keypoints.append((kp[m.trainIdx], kp_query[m.queryIdx],
                                   img_size_list[m.trainIdx], img_centroid_list[m.trainIdx],image_query_size))


# Make sure size and scale give similar results
test_size(matching_keypoints)
print("Number of good matches: ", len(matching_keypoints))

# cv2.drawMatchesKnn expects list of lists as matches.
# img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=2)
count = 0
# Apply hough transform
pose_bins = perform_hough_transform(matching_keypoints, 30, 2, 4)

# Get most voted
valid_bins = []  # A list of PoseBin objects
max_vote = 3
best_pose_bin = PoseBin()
dup_bins = []
for key in pose_bins:
    if pose_bins.get(key).votes >= 3:
        valid_bins.append(pose_bins.get(key))
        count += 1    # Added count to get length of valid_bins
    if pose_bins.get(key).votes > best_pose_bin.votes:
        best_pose_bin = pose_bins.get(key)
        dup_bins = [pose_bins.get(key)]
    elif pose_bins.get(key).votes == best_pose_bin.votes:
        dup_bins.append(pose_bins.get(key))


print("Number of duplicate votes: ", len(dup_bins))

img = cv2.drawKeypoints(gray_query, [kp[1] for kp in matching_keypoints], None, flags=4)
plt.imshow(img)

fig, ax = plt.subplots()
color_count = 0
colors = ['r', 'b', 'g', 'y']
for bin in dup_bins:
    print("Most Voted Pose: ", bin.pose, " with ", bin.votes, " votes")
    print("Box Size: ", bin.img_size, " in ", colors[color_count % len(colors)], "\n")
    ax = plot_rect(gray_query, bin, ax, colors[color_count % len(colors)])
    color_count += 1

plt.show()
print("main done")


#return valid_bins, count
