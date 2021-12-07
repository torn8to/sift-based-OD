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
from AffineParameters import *

#def main():
# Initiate SIFT detector
sift = cv2.SIFT_create()


image_query = cv2.imread('../Data_Set/3People_1Car.jpg')  # Query Image

rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
kp_query, des_query = sift.detectAndCompute(gray_query, None)
image_query_size = (len(gray_query[0]), len(gray_query))

img_size_list = []
img_centroid_list = []
with open('../Data_Set/training_data.pkl', 'rb') as inp:
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
                                   img_size_list[m.trainIdx], img_centroid_list[m.trainIdx], image_query_size))


# Make sure size and scale give similar results
test_size(matching_keypoints)
print("Number of good matches: ", len(matching_keypoints))

# cv2.drawMatchesKnn expects list of lists as matches.
# img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=2)
count = 0
# Apply hough transform
angle_factor = 10
scale_factor = 2
pos_factor = 32
pose_bins = perform_hough_transform(matching_keypoints, angle_factor, scale_factor, pos_factor)

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
plot_multiple_rect(gray_query, dup_bins, ax)

print("main done")


## Applying Affine parameters
update = True
num_updates = 0
while update:
    update = False
    max_vote = 3
    best_pose_bin = PoseBin()
    dup_bins = []
    remaining_bins = []
    for pose_bin in valid_bins:
        AffineParameters(pose_bin)  # Get Affine Parameters
        # Remove invalid keypoints
        pose_bin, change = remove_outliers(pose_bin, image_query_size, pos_factor*4, pos_factor*4)
        if change:  # if we changed the keypoints, run it again
            update = True
        if pose_bin.votes >= 3:  # Get a list of remaining valid bins
            remaining_bins.append(pose_bin)
            count += 1  # Added count to get length of valid_bins
        if pose_bin.votes > best_pose_bin.votes:  # Find which bins have the most votes
            best_pose_bin = pose_bin
            dup_bins = [pose_bin]
        elif pose_bin.votes == best_pose_bin.votes:  # store other bins with most votes
            dup_bins.append(pose_bin)
    valid_bins = remaining_bins  # remaining bins are valid
    num_updates += 1

print(num_updates)
fig, ax = plt.subplots()
plot_multiple_rect(gray_query, dup_bins, ax)
plt.show()

print("affine parameters done")
