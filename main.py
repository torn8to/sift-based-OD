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

image_query = cv2.imread('../CroppedImage.png')  # Query Image
rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
kp_query, des_query = sift.detectAndCompute(gray_query, None)

rgb_final_image = rgb_query
final_kp = kp_query
final_good = []

with open('training_data.pkl', 'rb') as inp:
    data = pickle.load(inp)
    for datum in data:
        datum[0] = make_kp(datum[0])
        img = cv2.imread(datum[2])

        # image comparison
        rgb_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the keypoints and descriptors with SIFT
        kp = datum[0]
        des = datum[1]

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_query, des, k=2)  # query ,database,nearest neighbors

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        # Every match has parameters
        # distance: the euclidean distance from the query descriptor to the training descriptor
        # imgIdx: Train image index
        # queryIdx: query descriptor index
        # trainIdx: train descriptor index

        if len(good) > len(final_good):
            final_good = good
            rgb_final_image = rgb_img1
            final_kp = kp
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(rgb_query, kp_query, rgb_final_image, final_kp, final_good[::100], None, flags=4)
    img4 = cv2.drawKeypoints(rgb_query, kp_query, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

plt.imshow(img3), plt.show()
