import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
from cv2 import sort
from SiftHelperFunctions import *

# Initiate SIFT detector
sift = cv2.SIFT_create()

image_query = cv2.imread('../Data_Set/IMG_20211027_170237_Rotated.jpg')  # Query Image
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
# TODO Will need to be changed for multiple model images
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
# TODO will need to be changed for multiple models
IMG_WIDTH = 1958
IMG_HEIGHT = 1575
angle_breakpoint = 30.0  # degrees
scale_breakpoint = 2.0

# Generate Pose guess of keypoints
pose_bins = {}
relaxed_bins = {}
for kpM, kpQ in matching_keypoints:
    octaveM, layerM, scaleM = unpack_sift_octave(kpM)  # unpack octave information for model keypoint
    octaveQ, layerQ, scaleQ = unpack_sift_octave(kpQ)  # unpack octave information for query keypoint

    # Changed from LOWE
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

    possible_scale = [scale_breakpoint ** np.floor(np.log(pose_estimate[3])/np.log(scale_breakpoint)),
                      scale_breakpoint ** np.ceil(np.log(pose_estimate[3])/np.log(scale_breakpoint))]

    for i in range(2):
        for j in range(2):
            for theta in range(2):
                try:
                    relaxed_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta])] += 1
                except:
                    relaxed_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta])] = 1

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

max_relaxed = (0,0,0,0)
max_relaxed_vote = 0
for key in relaxed_bins:
    if relaxed_bins.get(key) > max_relaxed_vote:
        print(relaxed_bins.get(key), key)
        max_relaxed = key
        max_relaxed_vote = relaxed_bins.get(key)
print(max_relaxed)

## VISUALIZATION ###############################################################
fig, ax = plt.subplots()
img = cv2.drawKeypoints(gray_query, queryImage_kp, None, None, flags=4)
plt.imshow(img)
# add box to image
x_shift = -IMG_WIDTH * max_pose[3] / 2
y_shift = -IMG_HEIGHT * max_pose[3] / 2

# Frame transformations to align box properly after rotation
# Denavit-Hartenberg parameters for 1st transform
theta1 = np.arctan2(max_pose[1], max_pose[0]) - np.pi
a1 = -np.sqrt(max_pose[0]**2 + max_pose[1]**2)

T01 = np.matrix([[np.cos(theta1), -np.sin(theta1), 0, a1*np.cos(theta1)],
                 [np.sin(theta1),  np.cos(theta1), 0, a1*np.sin(theta1)],
                 [0,               0,              1, 0],
                 [0,               0,              0, 1]])

# Denavit-Hartenberg Parameters for 2nd transform
theta2 = np.arctan2(y_shift, x_shift)-theta1
T12 = np.matrix([[np.cos(theta2), -np.sin(theta2), 0, 0],
                 [np.sin(theta2),  np.cos(theta2), 0, 0],
                 [0,               0,              1, 0],
                 [0,               0,              0, 1]])

# Location of left right corner with respect to center point of rectangle in that frame
pb_prime = np.matrix([[np.cos(np.deg2rad(max_pose[2]))*np.sqrt(x_shift**2 + y_shift**2)],
                      [np.sin(np.deg2rad(max_pose[2]))*np.sqrt(x_shift**2 + y_shift**2)], [0], [1]])

# matrix multiplication
T02 = np.matmul(T01, T12)
left_corner_pose = np.matmul(T02, pb_prime)
# End of Frame transformations

# simpler version of whats above
rect_left_corner = (max_pose[0] + np.cos(np.deg2rad(max_pose[2]))*x_shift - np.sin(np.deg2rad(max_pose[2]))*y_shift,
                    max_pose[1] + np.sin(np.deg2rad(max_pose[2]))*x_shift + np.cos(np.deg2rad(max_pose[2]))*y_shift)

rect = patches.Rectangle(rect_left_corner,
                         IMG_WIDTH * max_pose[3], IMG_HEIGHT * max_pose[3], max_pose[2],
                         linewidth=4, edgecolor='r', facecolor='none')
#TODO Rotation of box if image is rotated
ax.add_patch(rect)
plt.show()
print("done")
