# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
from cluster_hough_transform import *
import math

def make_kp(temp_kp):
    kp = []
    centroid = []
    for point in temp_kp:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2], response=point[3], octave=point[4], class_id=point[5])
        kp.append(temp)
        centroid.append(point[6])
    return kp, centroid


# Initiate SIFT detector
sift = cv2.SIFT_create()


image_query = cv2.imread('/home/prajwal/Desktop/cv_group_project/Sift-Implementation/Test_image.jpg')  # Query Image
print(len(image_query))
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
kp, centroid = make_kp(temp_kp)
des = temp_des

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_query, des, k=2)  # query ,database,nearest neighbors

# Apply ratio test
good_matches = []
model_kp = []
queryImage_kp = []
centroid_model = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
        model_kp.append(kp[m.trainIdx])
        centroid_model.append(centroid[m.trainIdx])
        queryImage_kp.append(kp_query[m.queryIdx])

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

hough_transform = []
bin_x = 10
bin_y = 10
bin_theta = 10
bin_sigma = 10

for i in range(bin_x):
    hough_transform.append([])
    for j in range(bin_y):
        hough_transform[i].append([])
        for k in range(bin_theta):
            hough_transform[i][j].append([])
            for l in range(bin_sigma):
                hough_transform[i][j][k].append(0)

for i in range(len(queryImage_kp)):
    q_kp = queryImage_kp[i]
    q_x = q_kp.pt[0]
    q_y = q_kp.pt[1]
    q_scale = q_kp.size
    q_theta = q_kp.angle
    for j in range(len(model_kp)):
        m_kp = model_kp[j]
        m_centroid = centroid_model[j]
        m_x = m_kp.pt[0]
        m_y = m_kp.pt[1]
        m_scale = m_kp.size
        m_theta = m_kp.angle
        #translation and scaling
        scale = q_scale / m_scale
        translated_x = (m_centroid[0] - m_x) * scale
        translated_y = (m_centroid[1] - m_y) * scale
        #rotation
        alpha = abs(m_theta - q_theta)
        rotated_x = math.cos(alpha) * translated_x - math.sin(alpha) * translated_y
        rotated_y = math.sin(alpha) * translated_x + math.cos(alpha) * translated_y
        #translate back to estimate the centroid of the query image
        q_centroid = (rotated_x + q_x, rotated_y + q_y)
        ##calculate x and y index for the hough transform bin
        i_x = (q_centroid[0] * bin_x) / img_width       ##img_width not initialized yet
        i_y = (q_centroid[1] * bin_y) / img_height      ##img_height not initialized yet
        ##normalize theta from [-pi, pi] to [0, 2*pi] and calculate index
        i_theta_prime = (alpha + math.pi) * bin_theta / (math.pi * 2)
        ##To allow for the rotations by −π and π to be close together
        i_theta = i_theta_prime % bin_theta
        ##determine the scale index
        i_sigma = (math.log(scale, base=2) / (2*(octaves - 1)) + 0.5) * bin_sigma    ###octaves not initialized yet
        for w in range(2):
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        hough_transform[i_x + w][i_y + x][i_theta + y][i_sigma + z] += 1



# cv2.drawMatchesKnn expects list of lists as matches.
img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=4)


plt.imshow(img), plt.show()
