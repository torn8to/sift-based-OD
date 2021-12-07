import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
from cv2 import sort
from SiftHelperFunctions import *
from VisualHelperFunctions import *
from HoughTransform import *
from PoseBin import *
from AffineParameters import *


# def main():
# Initiate SIFT detector

class Main:

    def __init__(self):
        self.kp = []
        self.des = []
        self.kp_query = []
        self.des_query = []
        self.gray_query = []
        self.img_size_list = []
        self.img_centroid_list = []
        self.image_query_size = (0, 0)

    def get_query_features(self):
        image_query = cv2.imread('../Test_image_14.jpeg')  # Query Image

        rgb_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2RGB)
        self.gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
        self.kp_query, self.des_query = sift.detectAndCompute(self.gray_query, None)
        self.image_query_size = (len(self.gray_query[0]), len(self.gray_query))

        img_size_list = []
        img_centroid_list = []
        with open('../Data_Set/training_data.pkl', 'rb') as inp:
            data = pickle.load(inp)  # Open training data and load it

            temp_kp = data[0][0]  # temporary kp, grab the first element in the data set
            self.img_size_list = [data[0][2]] * len(data[0][0])
            self.img_centroid_list = [data[0][3]] * len(data[0][0])
            temp_des = data[0][1]  # the descriptor vector, grab the first element in the data set
            for datum in data[1:]:  # for the remaining elements append them to the previous two lists
                temp_kp.extend(datum[0])  # have to use extend here, because we don't want cascading lists
                self.img_size_list.extend([datum[2]] * len(datum[0]))  # maintaining list of img_size for each keypoint
                self.img_centroid_list.extend([datum[3]] * len(datum[0]))  # maintain centroid for each keypoint
                temp_des = np.append(temp_des, datum[1], axis=0)  # for numpy vectors we append

        # Organize model key points and descriptors into single vector/matrix
        self.kp = make_kp(temp_kp)
        self.des = temp_des
        return

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
        self.matching_keypoints = []  # Tuples of (kpM, kpQ)
        for m, n in matches:
            if m.distance < 0.75 * n.distance or m.distance < 1.0:
                good_matches.append([m])
                # Store the matching keypoints in a tuple in a list
                self.matching_keypoints.append((self.kp[m.trainIdx], self.kp_query[m.queryIdx],
                                                self.img_size_list[m.trainIdx], self.img_centroid_list[m.trainIdx],
                                                self.image_query_size))
        # Make sure size and scale give similar results
        test_size(self.matching_keypoints)
        print("Number of good matches: ", len(self.matching_keypoints))
        return self.matching_keypoints


    def apply_hough_transform(self):
        # cv2.drawMatchesKnn expects list of lists as matches.
        # img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=2)
        # Apply hough transform
        angle_factor = 10
        scale_factor = 2
        pos_factor = 32
        pose_bins = perform_hough_transform(self.matching_keypoints, angle_factor, scale_factor, pos_factor)

        # Get most voted
        valid_bins = []  # A list of PoseBin objects
        best_pose_bin = PoseBin()
        best_pose_bin.votes = 3
        dup_bins = []
        for key in pose_bins:
            if pose_bins.get(key).votes >= 3:
                valid_bins.append(pose_bins.get(key))
            if pose_bins.get(key).votes > best_pose_bin.votes:
                best_pose_bin = pose_bins.get(key)
                dup_bins = [pose_bins.get(key)]
            elif pose_bins.get(key).votes == best_pose_bin.votes:
                dup_bins.append(pose_bins.get(key))

        print("Number of duplicate votes: ", len(dup_bins))

        img = cv2.drawKeypoints(self.gray_query, [kp[1] for kp in self.matching_keypoints], None, flags=4)
        plt.imshow(img)

        fig, ax = plt.subplots()
        plot_multiple_rect(self.gray_query, dup_bins, ax)

        print("main done")

        ## Applying Affine parameters
        update = True
        num_updates = 0
        while update:
            update = False
            max_vote = 3
            best_pose_bin = PoseBin()
            dup_bins = []
            remaining_bins = []
            for pose_bin in valid_bins:
                AffineParameters(pose_bin)  # Get Affine Parameters
                # Remove invalid keypoints
                _, change = remove_outliers(pose_bin, self.image_query_size, pos_factor * 4, pos_factor * 4)
                if change:  # if we changed the keypoints, run it again
                    update = True
                if pose_bin.votes >= 3:  # Get a list of remaining valid bins
                    remaining_bins.append(pose_bin)
                if pose_bin.votes > best_pose_bin.votes:  # Find which bins have the most votes
                    best_pose_bin = pose_bin
                    dup_bins = [pose_bin]
                elif pose_bin.votes == best_pose_bin.votes:  # store other bins with most votes
                    dup_bins.append(pose_bin)
            valid_bins = remaining_bins  # remaining bins are valid
            num_updates += 1

        print(num_updates)
        fig, ax = plt.subplots()
        plot_multiple_rect(self.gray_query, dup_bins, ax)
        plt.show()

        print("affine parameters done")


if __name__ == "__main__":
    sift = cv2.SIFT_create()
    main = Main()
    main.get_query_features()
    main.run_matcher()
    main.apply_hough_transform()
    main.apply_affine_parameters()
