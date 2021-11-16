# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
import math

# from numpy.core.fromnumeric import shape

def unpack_sift_octave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)
    return (octave, layer, scale)

def make_kp(temp_kp):
    kp = []
    centroid = []
    shape = []
    for point in temp_kp:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2], response=point[3], octave=point[4], class_id=point[5])
        kp.append(temp)
        centroid.append(point[6])
        shape.append(point[7])
    return kp, centroid, shape


# Initiate SIFT detector
sift = cv2.SIFT_create()


image_query = cv2.imread('/home/prajwal/Desktop/cv_group_project/Sift-Implementation/Test_image5.jpeg')  # Query Image
rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
kp_query, des_query = sift.detectAndCompute(gray_query, None)

with open('training_data.pkl', 'rb') as inp:
    data = pickle.load(inp)

    temp_kp = data[0][0]
    temp_des = data[0][1]
    for datum in data[1:]:
        temp_kp.extend(datum[0])
        temp_des = np.append(temp_des, datum[1],axis=0)

        # Unecessary Code for visualization
        # img = cv2.imread(datum[2])
        #
        # # image comparison
        # rgb_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Organize model key points and descriptors into single vector/matrix
kp, centroid, shape = make_kp(temp_kp)
des = temp_des

# Calculate maximum octave
max_octave = 0
for keypoint in kp:
    octave, _, _ = unpack_sift_octave(keypoint)
    max_octave = max(max_octave, octave)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_query, des, k=2)  # query ,database,nearest neighbors

# Apply ratio test
good_matches = []
model_kp = []
shape_model_kp = []  ##stores the information of shape image from which the keypoint was generated
queryImage_kp = []
centroid_model = []  ##stores the centroid information corresponding to image keypoint
matching_keypoints = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
        model_kp.append(kp[m.trainIdx])
        shape_model_kp.append(shape[m.trainIdx])
        centroid_model.append(centroid[m.trainIdx])
        queryImage_kp.append(kp_query[m.queryIdx])
        matching_keypoints.append((kp[m.trainIdx], kp_query[m.queryIdx]))

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

hough_transform = []
img_height = image_query.shape[0]
img_width = image_query.shape[1]
# print(img_height)
# print(img_width)
# octaves = 128
bin_x = 10
bin_y = 10
bin_theta = 10
bin_sigma = 10

##Initialize 4 dimensional array for estimating the centroid of object 

for i in range(bin_x):
    hough_transform.append([])
    for j in range(bin_y):
        hough_transform[i].append([])
        for k in range(bin_theta):
            hough_transform[i][j].append([])
            for l in range(bin_sigma):
                hough_transform[i][j][k].append([])


for index, element in enumerate(matching_keypoints):
    kpM = element[0]
    kpQ = element[1]
    q_x = kpQ.pt[0]
    q_y = kpQ.pt[1]
    q_octave, q_layer, q_scale = unpack_sift_octave(kpQ)
    q_theta = kpQ.angle
    m_centroid = centroid_model[index]
    m_x = kpM.pt[0]
    m_y = kpM.pt[1]
    m_octave, m_layer, m_scale = unpack_sift_octave(kpM)
    m_theta = kpM.angle
    #translation and scaling
    # scale = q_scale / m_scale
    scale = kpQ.size / kpM.size
    translated_x = (m_centroid[0] - m_x) * scale
    translated_y = (m_centroid[1] - m_y) * scale
    #rotation
    alpha = abs(m_theta - q_theta)
    rotated_x = math.cos(alpha) * translated_x - math.sin(alpha) * translated_y
    rotated_y = math.sin(alpha) * translated_x + math.cos(alpha) * translated_y
    #translate back to estimate the centroid of the query image
    q_centroid = (rotated_x + q_x, rotated_y + q_y)
    ##calculate x and y index for the hough transform bin
    i_x = int((q_centroid[0] * bin_x) / img_width)     
    i_x = max(0, i_x - 1)                                ## making sure the index does not go out of range
    i_x = min(i_x, bin_x - 1)                            ## making sure the index does not go out of range
    i_y = int((q_centroid[1] * bin_y) / img_height)  
    i_y = max(0, i_y- 1)                                 ## making sure the index does not go out of range
    i_y = min(i_y, bin_y - 1)                           ## making sure the index does not go out of range
    ##normalize theta from [-pi, pi] to [0, 2*pi] and calculate index
    i_theta_prime = (alpha + math.pi) * bin_theta / (math.pi * 2)
    ##To allow for the rotations by -pi or pi to be close together
    i_theta = int(i_theta_prime % bin_theta)
    ##determine the scale index
    n_oct = 4 ##########Assuming number of octaves as 4
    i_sigma = int((math.log(scale, 2) / (2 * (max_octave - 1)) + 0.5) * bin_sigma)
    i_sigma = max(0, i_sigma)                           ## making sure the index does not go out of range
    i_sigma = min(i_sigma, bin_sigma - 1)               ## making sure the index does not go out of range
    for w in range(2):
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    a = i_x + w
                    b = i_y + x
                    c = i_theta + y
                    d = i_sigma + z
                    if (a < bin_x and b < bin_y and c < bin_theta and d < bin_sigma):
                            hough_transform[a][b][c][d].append(element)
                            # hough_transform[a][b][c][d] +=1
                        # print(i_x +w, i_y +x, i_theta + y, i_sigma + z)
print("Matches before Hough Transform:")
print(len(good_matches))
# c= 0
# for i in range(bin_x):
#     for j in range(bin_y):
#         for k in range(bin_theta):
#             for l in range(bin_sigma):
#                 if len(hough_transform[i][j][k][l]) >= 5:
                    # print(i, j, k, l)
                    # c+=1
                    # print(len(hough_transform[i][j][k][l]))
# print(c)
count = 0
best_matches_kp_query = []
best_matches_kp_model = []
best_pose = []
for i in range(bin_x):
    for j in range(bin_y):
        for k in range(bin_theta):
            for l in range(bin_sigma):
                if len(hough_transform[i][j][k][l]) >= 3:
                    count += 1
                    best_pose.append((i, j))
                    # print(len(hough_transform[i][j][k][l]))
                    # element = hough_transform[i][j][k][l]
                    for element in hough_transform[i][j][k][l]:
                        best_matches_kp_query.append(element[1])
                        best_matches_kp_model.append(element[0])


print("Total Estimated object centroids after Hough Transform:")
print(count)
print("Total keypoints after Hough Transform")
print(len(best_matches_kp_query))
# # cv2.drawMatchesKnn expects list of lists as matches.
# img = cv2.drawKeypoints(rgb_query, best_matches_kp_query, None, flags=4)


# plt.imshow(img), plt.show()

## VISUALIZATION ###############################################################

fig, ax = plt.subplots()
img = cv2.drawKeypoints(rgb_query, best_matches_kp_query, None, flags=4)
# img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=4)
plt.imshow(img)
# add box to image
temp = 0.05
for pose in best_pose:
    x_c = pose[0]*img_width / bin_x
    y_c = pose[1]*img_height / bin_y
    rect_left_corner = (max(x_c - img_width * temp / 2, 0),
                    max(y_c - img_height * temp / 2, 0))

    rect = patches.Rectangle(rect_left_corner,
                         img_width * temp, img_height * temp, 0,
                         linewidth=3, edgecolor='r', facecolor='none')
#TODO Rotation of box if image is rotated
    ax.add_patch(rect)
plt.show()
print("done")