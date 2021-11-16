import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
from cv2 import sort
from SiftHelperFunctions import *

# Initiate SIFT detector
sift = cv2.SIFT_create()

image_query = cv2.imread('../Data_Set/IMG_Standing_Rotated.jpg')  # Query Image
rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
kp_query, des_query = sift.detectAndCompute(gray_query, None)

img_size_list = []
img_centroid_list = []
with open('training_data.pkl', 'rb') as inp:
    data = pickle.load(inp)  # Open training data and load it

    temp_kp = data[0][0]  # temporary kp, grab the first element in the data set
    img_size_list = [data[0][2]] * len(data[0][0])
    img_centroid_list = [data[0][3]]*len(data[0][0])
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
queryImage_kp = []
matching_keypoints = []  # Tuples of (kpM, kpQ)
for m, n in matches:
    if m.distance < 0.75 * n.distance or m.distance < 1.0:
        good_matches.append([m])
        # Store the matching keypoints in a tuple in a list
        queryImage_kp.append(kp_query[m.queryIdx])
        matching_keypoints.append((kp[m.trainIdx], kp_query[m.queryIdx],
                                   img_size_list[m.trainIdx], img_centroid_list[m.trainIdx]))

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

# cv2.drawMatchesKnn expects list of lists as matches.
# img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=2)


# INITIAL VALUES
angle_breakpoint = 30.0  # degrees
scale_breakpoint = 2.0

# Generate Pose guess of keypoints
pose_bins = {}
for kpM, kpQ, img_size, img_centroid in matching_keypoints:

    octaveM, layerM, scaleM = unpack_sift_octave(kpM)  # unpack octave information for model keypoint
    octaveQ, layerQ, scaleQ = unpack_sift_octave(kpQ)  # unpack octave information for query keypoint

    # Changed from LOWE
    x_pos_breakpoint = img_size[0] * scaleQ / 32.0  # determine x axis bucket size in pixels
    y_pos_breakpoint = img_size[1] * scaleQ / 32.0  # determine y axis bucket size in pixels

    pose_estimate = (0, 0, 0, 0)  # Pose consists of x,y,orientation,scale for the centroid of the object

    scale_diff = scaleM / scaleQ
    x_diff = kpQ.pt[0] - kpM.pt[0]
    y_diff = kpQ.pt[1] - kpM.pt[1]
    orientation_diff = normalize_angle(kpQ.angle - kpM.angle)

    pose_estimate = (img_centroid[0] + x_diff, img_centroid[1] + y_diff, orientation_diff, scale_diff)

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
                for s in range(2):
                    try:
                        """
                        We first update the width and height of our image using the average of all widths
                        and heights used
                        """
                        count = pose_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta],
                                           possible_scale[s])][0]
                        old_width = pose_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta],
                                               possible_scale[s])][1][0]
                        old_height = pose_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta],
                                                possible_scale[s])][1][1]

                        new_width = (old_width * count + img_size[0]) / (count + 1)
                        new_height = (old_height * count + img_size[1]) / (count + 1)

                        """
                        Then we actually update the vote
                        """
                        pose_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta],
                                   possible_scale[s])][0] += 1
                        pose_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta],
                                   possible_scale[s])][1] = (new_width, new_height)

                    except KeyError:
                        pose_bins[(possible_x_pos[i], possible_y_pos[j], possible_orientation[theta],
                                   possible_scale[s])] = [1, img_size]

max_pose = (0, 0, 0, 0)
des_img_size = (0, 0)
max_vote = 0
for key in pose_bins:
    if pose_bins.get(key)[0] > max_vote:
        print("Number of votes: ", pose_bins.get(key)[0], "for pose ", key)
        max_pose = key
        des_img_size = pose_bins.get(key)[1]
        max_vote = pose_bins.get(key)[0]
print("Most Voted Pose: ", max_pose)
print("Box Size: ", des_img_size)

## VISUALIZATION ###############################################################
fig, ax = plt.subplots()
img = cv2.drawKeypoints(gray_query, queryImage_kp, None, None, flags=4)
plt.imshow(img)
# add box to image
IMG_WIDTH = des_img_size[0]
IMG_HEIGHT = des_img_size[1]
x_shift = -IMG_WIDTH * max_pose[3] / 2
y_shift = -IMG_HEIGHT * max_pose[3] / 2

# simpler version of whats above
rect_left_corner = (max_pose[0] + np.cos(np.deg2rad(max_pose[2]))*x_shift - np.sin(np.deg2rad(max_pose[2]))*y_shift,
                    max_pose[1] + np.sin(np.deg2rad(max_pose[2]))*x_shift + np.cos(np.deg2rad(max_pose[2]))*y_shift)

rect = patches.Rectangle(rect_left_corner,
                         IMG_WIDTH * max_pose[3], IMG_HEIGHT * max_pose[3], max_pose[2],
                         linewidth=4, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()
print("done")
