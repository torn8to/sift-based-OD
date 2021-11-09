# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle


def make_kp(temp_kp):
    kp = []
    for point in temp_kp:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                            _response=point[3], _octave=point[4], _class_id=point[5])
        kp.append(temp)
    return kp


# Initiate SIFT detector
sift = cv2.SIFT_create()

image_query = cv2.imread('../EditedImage.png')  # Query Image
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
kp = make_kp(temp_kp)
des = temp_des

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_query, des, k=2)  # query ,database,nearest neighbors

# Apply ratio test
good_matches = []
model_kp = []
queryImage_kp = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
        model_kp.append(kp[m.trainIdx])
        queryImage_kp.append(kp_query[m.queryIdx])

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

# cv2.drawMatchesKnn expects list of lists as matches.
img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=4)


plt.imshow(img), plt.show()
