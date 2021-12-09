# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
from SiftHelperFunctions import *
from PoseBin import *
from HoughTransformHelperFunctions import *
from AffineParameters import *
from PostProcessing import *
from VisualHelperFunctions import *


# def main():
# Initiate SIFT detector

class Main:

    def __init__(self):
        self.kp = []
        self.des = []
        self.kp_query = []
        self.des_query = []
        self.rgb_query = []
        self.gray_query = []
        self.img_size_list = []
        self.img_centroid_list = []
        self.image_query_size = (0, 0)
        self.matching_keypoints = []  # Tuples of matching (Model keypoints, Query keypoints)
        self.hough_transform = {}     # Dictionary for storing posebin objects to estimate centroid
        self.valid_bins = []          # List of all hough transform bins having votes greater than threshold
        self.keypoint_pairs = []      # List of all keypoints pairs for the hough transform bin with votes greater than threshold
        self.final_pose = []
        _, self.ax = plt.subplots()

    def get_query_features(self, path):
        image_query = cv2.imread(path)
        # Get the image in RGB scale
        self.rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
        # Get the image in gray scale
        self.gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
        # Identify sift keypoints and descriptors
        self.kp_query, self.des_query = sift.detectAndCompute(self.gray_query, None)     
        # Get the image size
        self.image_query_size = (len(self.gray_query[0]), len(self.gray_query))
        # self.image_query_size = (len(self.gray_query[0]), len(self.gray_query))   

        training_data_path = '/home/prajwal/Desktop/cv_group_project/Data_Set/Train_DataSet/DatabaseInfo/standing/training_data.pkl'
        # open the training data and decipher input
        with open(training_data_path, 'rb') as inp:
            data = pickle.load(inp)  # Open training data and load it

            temp_kp = data[0][0] # temporary kp, grab the first element in the data set
            self.img_size_list = [data[0][2]] * len(data[0][0]) # keep track of size of each model image
            self.img_centroid_list = [data[0][3]] * len(data[0][0]) # keep track of centroid of each model image
            temp_des = data[0][1]  # the descriptor vector, grab the first element in the data set
            for datum in data[1:]: # for the remaining elements append them to the previous two lists
                temp_kp.extend(datum[0]) # have to use extend here, because we don't want cascading lists
                self.img_size_list.extend([datum[2]] * len(datum[0]))  # maintaining list of img_size for each keypoint
                self.img_centroid_list.extend([datum[3]] * len(datum[0]))  # maintain centroid for each keypoint
                temp_des = np.append(temp_des, datum[1],axis=0) # for numpy vectors we append


        # Organize model key points and descriptors into single vector/matrix
        self.kp = make_kp(temp_kp)
        self.des = temp_des

    def run_matcher(self):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(self.des_query, self.des, k=2)  # query ,database,nearest neighbors

        # Every match has parameters
        # distance: the euclidean distance from the query descriptor to the training descriptor
        # imgIdx: Train image index
        # queryIdx: query descriptor index
        # trainIdx: train descriptor index

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
                # Store the matching keypoints in a tuple in a list - (model keypoint, query_keypoint, model image size, model centroid)
                self.matching_keypoints.append((self.kp[m.trainIdx], self.kp_query[m.queryIdx],  
                                            self.img_size_list[m.trainIdx], self.img_centroid_list[m.trainIdx]))
                                            #self.image_query_size

    def apply_hough_transform(self, bins=15):

        for tuple_list in self.matching_keypoints:
            # Get matching keypoint pair (kpM, kpQ)
            keypoint_pair = tuple_list[0], tuple_list[1]

            # Get model image size
            model_image_size = tuple_list[2]

            # Estimate the object pose in query image
            object_pose = estimate_object_pose(tuple_list)

            # calculate bin indices
            i_x, i_y, i_theta, i_sigma = calculate_bin_index(object_pose, bins, self.rgb_query.shape)
         
            # Add votes for two adjacent bins (i and i+1) to account for discretization errors
            for w in range(2):
                for x in range(2):
                    for y in range(2):
                        for z in range(2):
                            pose = (i_x + w, i_y + x, i_theta + y, i_sigma + z)
                            if (pose[0] < bins and pose[1] < bins and pose[2] < bins and pose[3] < bins):
                                try:
                                    # update the posebin object with new object pose, image size and keypoint pair
                                    self.hough_transform[pose].update_posebin(object_pose, model_image_size, keypoint_pair)
                                except KeyError:
                                    # since this is the first object in this bin, the mean for this bin 
                                    # will be equal to estimated centroid, orientation and scale,
                                    mean = object_pose
                                    # append the posebin object into the bin
                                    self.hough_transform[pose] = PoseBin(pose, model_image_size, 1, [keypoint_pair], mean)

    def get_valid_bins(self, threshold=5):
        
        self.keypoint_pairs = []
        keys = self.hough_transform.keys()

        for key in keys:
            if self.hough_transform[key].votes >= threshold:
                # store all keypoint pairs as a list
                self.keypoint_pairs.extend(self.hough_transform[key].keypoint_pairs)
                # Get rid of duplicates
                if self.hough_transform[key] not in self.valid_bins:
                    self.valid_bins.append(self.hough_transform[key])

    def update_keypoint_pairs(self):
        self.keypoint_pairs.clear()
        for posebin in self.valid_bins:
            self.keypoint_pairs.extend(posebin.keypoint_pairs)

    def apply_affine_parameters(self, threshold):
        # Applying Affine parameters
        pos_factor=32
        update = True
        num_updates = 0
        while update:  # While we are eliminating outliers keep applying affine parameters
            update = False
            remaining_bins = []
            for pose_bin in self.valid_bins:
                AffineParameters(pose_bin)  # Get Affine Parameters
                # Remove invalid keypoints
                _, change = remove_outliers(pose_bin, self.image_query_size, pos_factor * 4, pos_factor * 4)
                update = change or update  # if we changed the keypoints, set flag to true
                if pose_bin.votes >= threshold:  # Get a list of remaining valid bins
                    remaining_bins.append(pose_bin)
            self.valid_bins = remaining_bins  # remaining bins are valid
            num_updates += 1
        # end while loop
        self.update_keypoint_pairs()

    def post_process(self):
         ## Group all nearby best poses based on x, y coordinates
        pose_cluster = group_position(self.valid_bins)
        ##Group all nearby orientations for each pose cluster
        orientation_cluster = group_orientation(pose_cluster)
        # Select the most dominant orientation for each pose cluster
        final_orientation_list = find_max_orientation(orientation_cluster)
        ## Final pose is the average of pose_cluster position, maximum of orienation cluster for each position,
        #  average scale for each pose_cluster and minimum image area for each pose_cluster
        self.final_pose = get_final_pose(pose_cluster, final_orientation_list)

   

    def plot(self):
        # plots keypoints and draws rectangle around it
        self.ax = show_keypoints(self.rgb_query, self.keypoint_pairs, self.ax)
        # draws rectangle around the object
        self.ax = show_object(self.final_pose, self.ax)

if __name__ == "__main__":
    sift = cv2.SIFT_create()
    main = Main()
    main.get_query_features('/home/prajwal/Desktop/cv_group_project/Data_Set/Test_DataSet/other/Test_image_09.jpg')
    main.run_matcher()
    main.apply_hough_transform(15) # Applies hough transform with 15x15x15x15 array for x, y, theta, sigma
    main.get_valid_bins(5)      # Gets bins having votes >= 5
    main.apply_affine_parameters(4) # Gets bins with votes >= 5 after affine transformation
    main.post_process()
    main.plot()
    plt.show()

