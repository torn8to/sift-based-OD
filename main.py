import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
from cv2 import sort
from SiftHelperFunctions import *

# Initiate SIFT detector
sift = cv2.SIFT_create()

image_query = cv2.imread('../Data_Set/IMG_20211027_170135.jpg')  # Query Image
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
    max_octave = max(max_octave,octave)
    x, y = keypoint.pt
    new_x = (centroid[0] * count + x) / (count + 1)
    new_y = (centroid[1] * count + y) / (count + 1)
    centroid = (new_x, new_y)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_query, des, k=2)  # query ,database,nearest neighbors

# Apply ratio test
good_matches = []
queryImage_kp = []
matching_keypoints = []
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

# Plot keypoints for largest bin
plot_kp = []
for angle_diff, index in bins[best_bin_index]:
    plot_kp.append(matching_keypoints[index][1])
img = cv2.drawKeypoints(gray_query, plot_kp, None, flags=4)
plt.imshow(img), plt.show()

# Generate Pose guess of keypoints
angle_breakpoint = 30  # degrees

x_pos_breakpoint = int(3000/4)
y_pos_breakpoint = int(4000/4)

pose_bins = {}
for kpM, kpQ in matching_keypoints:
    octaveM, layerM, scaleM = unpack_sift_octave(kpM)
    octaveQ, layerQ, scaleQ = unpack_sift_octave(kpQ)
    pose_estimate = (0, 0, 0, 0)  # Pose consists of x,y,orientation,scale for the centroid of the object

    x_diff = kpM.pt[0] - kpQ.pt[0]
    y_diff = kpM.pt[1] - kpQ.pt[1]
    orientation_diff = normalize_angle(kpM.angle - kpQ.angle)
    scale_diff = scaleM - scaleQ

    pose_estimate = (centroid[0] - x_diff, centroid[1] - y_diff, orientation_diff, 0)
    for i in range(4):
        for j in range(2):
            try:
                pose_bins[pose_estimate] += 1
            except:
                pose_bins[pose_estimate] = 1

print("done")
