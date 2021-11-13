import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
from cv2 import sort
from SiftHelperFunctions import *

# Initiate SIFT detector
sift = cv2.SIFT_create()

image_query = cv2.imread('../Data_Set/IMG_20211027_170237.jpg')  # Query Image
rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
kp_query, des_query = sift.detectAndCompute(gray_query, None)

with open('training_data.pkl', 'rb') as inp:
    data = pickle.load(inp)  # Open training data and load it

    temp_kp = data[0][0]  # temporary kp, grab the first element in the data set
    temp_des = data[0][1]  # the descriptor vector, grab the first elment in the data set
    for datum in data[1:]:  # for the remaining elements append them to the previous two lists
        temp_kp.extend(datum[0])  # have to use extend here, because we don't want cascading lists
        temp_des = np.append(temp_des, datum[1], axis=0)  # for numpy vectors we append

# Organize model key points and descriptors into single vector/matrix
kp = make_kp(temp_kp)
des = temp_des

# Create a centroid from the model keypoint positions
centroid = (0, 0)
count = 0
max_octave = 0
for keypoint in kp:
    octave, _, _ = unpack_sift_octave(keypoint)
    max_octave = max(max_octave, octave)
    x, y = keypoint.pt

    # Just averaging x and y positions
    new_x = (centroid[0] * count + x) / (count + 1)
    new_y = (centroid[1] * count + y) / (count + 1)
    centroid = (new_x, new_y)
    count += 1

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_query, des, k=2)  # query ,database,nearest neighbors

# Apply ratio test
good_matches = []
queryImage_kp = []
matching_keypoints = []  # Tuples of (kpM, kpQ)
for m, n in matches:
    if m.distance < 0.75 * n.distance or m.distance < 1.0:
        good_matches.append([m])
        # Store the matching keypoints in a tuple in a list
        queryImage_kp.append(kp_query[m.queryIdx])
        matching_keypoints.append((kp[m.trainIdx], kp_query[m.queryIdx]))

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

# cv2.drawMatchesKnn expects list of lists as matches.
# img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=2)


# INITIAL VALUES
IMG_WIDTH = 1958
IMG_HEIGHT = 1575
angle_breakpoint = 30.0  # degrees
scale_breakpoint = 2.0

# Generate Pose guess of keypoints
pose_bins = {}
for kpM, kpQ in matching_keypoints:
    octaveM, layerM, scaleM = unpack_sift_octave(kpM)  # unpack octave information for model keypoint
    octaveQ, layerQ, scaleQ = unpack_sift_octave(kpQ)  # unpack octave information for query keypoint
    x_pos_breakpoint = IMG_WIDTH * scaleQ / 32.0  # determine x axis bucket size in pixels
    y_pos_breakpoint = IMG_HEIGHT * scaleQ / 32.0  # determine y axis bucket size in pixels

    pose_estimate = (0, 0, 0, 0)  # Pose consists of x,y,orientation,scale for the centroid of the object

    scale_diff = scaleM / scaleQ
    x_diff = kpQ.pt[0] - kpM.pt[0]
    y_diff = kpQ.pt[1] - kpM.pt[1]
    orientation_diff = normalize_angle(kpQ.angle - kpM.angle)

    pose_estimate = (centroid[0] + x_diff, centroid[1] + y_diff, orientation_diff, scale_diff)

    # Get bucket locations
    possible_x_pos = [int(np.floor(pose_estimate[0] / x_pos_breakpoint) * x_pos_breakpoint),
                      int(np.ceil(pose_estimate[0] / x_pos_breakpoint) * x_pos_breakpoint)]
    possible_y_pos = [int(np.floor(pose_estimate[1] / y_pos_breakpoint) * y_pos_breakpoint),
                      int(np.ceil(pose_estimate[1] / y_pos_breakpoint) * y_pos_breakpoint)]
    possible_orientation = [int(np.floor(pose_estimate[2] / angle_breakpoint) * angle_breakpoint),
                            int(np.ceil(pose_estimate[2] / angle_breakpoint) * angle_breakpoint)]
    possible_scale = [2 ** np.floor(np.log2(pose_estimate[3])),
                      2 ** np.ceil(np.log2(pose_estimate[3]))]

    for i in range(2):
        for j in range(2):
            for theta in range(2):
                for s in range(2):
                    try:
                        pose_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta],
                                   possible_scale[s])] += 1
                    except:
                        pose_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta],
                                   possible_scale[s])] = 1

max_pose = (0, 0, 0, 0)
max_vote = 0
for key in pose_bins:
    if pose_bins.get(key) > max_vote:
        print(pose_bins.get(key), key)
        max_pose = key
        max_vote = pose_bins.get(key)

print(max_pose)

## VISUALIZATION ###############################################################
fig, ax = plt.subplots()
img = cv2.drawKeypoints(gray_query, kp_query, None, None, flags=4)
plt.imshow(img)
# add box to image
rect_left_corner = (max(max_pose[0] - IMG_WIDTH * max_pose[3] / 2, 0),
                    max(max_pose[1] - IMG_HEIGHT * max_pose[3] / 2, 0))

rect = patches.Rectangle(rect_left_corner,
                         IMG_WIDTH * max_pose[3], IMG_HEIGHT * max_pose[3], 0,
                         linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()
print("done")
