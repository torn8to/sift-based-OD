import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
from cv2 import sort
from SiftHelperFunctions import *

# Initiate SIFT detector
sift = cv2.SIFT_create()


image_query = cv2.imread('../RandomImage.jpg')  # Query Image
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


index_list = []  # Needed only if the list is not sorted based on index

count = 0
delta = [] # list containing tuples of angle differences and the index to the matching keypoints
# Determine the angle difference between matching keypoints
for train_kp, query_kp in matching_keypoints:
    value_tuple = (normalize_angle(train_kp.angle - query_kp.angle), count)
    delta.append(value_tuple)
    count += 1

''' 
Calculate the difference in angles for each entry and form bins
Threshold will be +- 20 degrees
Form a 2D array to achieve this
We need to keepn track of the angle and the count
'''

# Create bins
num_bins = 12  # Determine the number of bins we want to use
angle_breakpoint = 360/num_bins  # break the angle cut offs based on the number of bins we have
bins = [[] for x in range(num_bins)]  # Initialize the bins to be empty arrays
for angle_diff, index in delta:  # Populate each bin
    # The bin number is just the integer of the angle difference divided by the angle breakpoint with an added offset to
    # make sure the values are between 0 and num_bins instead of -num_bins/2 and num_bins/2
    bin_number = int(angle_diff/angle_breakpoint + num_bins/2)
    # populate the bin with the angle difference and the index from matching_keypoints
    bins[bin_number].append((angle_diff, index))


votes = [len(b) for b in bins]  # tally up the number of elements in each bin

best_bin_count = max(votes)
best_bin_index = votes.index(best_bin_count)
print(best_bin_index)

# Display histogram of delta data
plt.hist([x[0] for x in delta], bins=num_bins)
plt.ylabel('Num votes')
plt.xlabel('bins')
plt.show()

# Plot keypoints for largest bin
plot_kp = []
for angle_diff, index in bins[best_bin_index]:
    plot_kp.append(matching_keypoints[index][1])
img = cv2.drawKeypoints(gray_query, plot_kp, None, flags=4)
plt.imshow(img), plt.show()
