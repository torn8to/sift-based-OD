import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
from cv2 import sort
from SiftHelperFunctions import *

# Initiate SIFT detector
sift = cv2.SIFT_create()


image_query = cv2.imread('Car8.jpg')  # Query Image
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
modelImage_kp = []
queryImage_kp = []
for m, n in matches:
    if m.distance < 0.75 * n.distance or m.distance < 1.0:
        good_matches.append([m])
        modelImage_kp.append(kp[m.trainIdx])
        queryImage_kp.append(kp_query[m.queryIdx])

# Every match has parameters
# distance: the euclidean distance from the query descriptor to the training descriptor
# imgIdx: Train image index
# queryIdx: query descriptor index
# trainIdx: train descriptor index

# cv2.drawMatchesKnn expects list of lists as matches.
img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=2)

srcImage_angle = []
modelImage_angle = []
index_list = []  # Needed only if the list is not sorted based on index
delta = []
for i in queryImage_kp:
    srcImage_angle.append(i.angle)
    

for i in modelImage_kp:
    modelImage_angle.append(i.angle)

''' 
Calculate the difference in angles for each entry and form bins
Threshold will be +- 20 degrees
Form a 2D array to achieve this
We need to keepn track of the angle and the count
'''

delta = [modelImage_angle - srcImage_angle for modelImage_angle, srcImage_angle in zip(modelImage_angle, srcImage_angle)]
delta.sort()

# Create bins

bin_1 = []
bin_2 = []
bin_3 = []
bin_4 = []
bin_5 = []
bin_6 = []
bin_7 = []
bin_8 = []
bin_9 = []
bin_10 = []
bin_11 = []
bin_12 = []
count = []
for i in delta:
    if (-180 < i <= -150):
        bin_1.append(i)
    if (-150 < i <= -120):
        bin_2.append(i)
    if (-120 < i <= -90):
        bin_3.append(i)
    if (-90 < i <= -60):
        bin_4.append(i)
    if (-60 < i <= -30):
        bin_5.append(i)
    if (-30 < i <= 0):
        bin_6.append(i)
    if (0 < i <= 30):
        bin_7.append(i)
    if (30 < i <= 60):
        bin_8.append(i)
    if (60 < i <= 90):
        bin_9.append(i)
    if (90 < i <= 120):
        bin_10.append(i)
    if (120 < i <= 150):
        bin_11.append(i)
    if (150 < i <= 180):
        bin_12.append(i)

votes = [len(bin_1), len(bin_2), len(bin_3), len(bin_4), len(bin_5), len(bin_6), len(bin_7), len(bin_8), len(bin_9), len(bin_10), len(bin_11), len(bin_12)]


best_bin_count = max(votes)
best_bin_index = votes.index(best_bin_count)
print(best_bin_index)

plt.imshow(img), plt.show()
