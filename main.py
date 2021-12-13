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
        self.valid_bins = []  # A list of PoseBin objects
        self.matching_keypoints = []  # Tuples of (kpM, kpQ)
        self.kp = []
        self.des = []
        self.kp_query = []
        self.des_query = []
        self.gray_query = []
        self.img_size_list = []
        self.img_centroid_list = []
        self.image_query_size = (0, 0)
        self.ax = plt.axes()

    def get_query_features(self, path):
        image_query = cv2.imread(path)  # Query Image

        # Get image in gray scale
        self.gray_query = cv2.cvtColor(image_query, cv2.COLOR_BGR2GRAY)
        # identify sift keypoints and descriptors
        self.kp_query, self.des_query = sift.detectAndCompute(self.gray_query, None)
        # Get the image size
        self.image_query_size = (len(self.gray_query[0]), len(self.gray_query))

        # open the training data and decipher input
        with open('../Data_Set/training_data.pkl', 'rb') as inp:
            data = pickle.load(inp)  # Open training data and load it

            temp_kp = data[0][0]  # temporary kp, grab the first element in the data set
            self.img_size_list = [data[0][2]] * len(data[0][0])  # keep track of size of each model image
            self.img_centroid_list = [data[0][3]] * len(data[0][0])  # keep track of centroid of each model image
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
        for m, n in matches:
            # print(m.distance)
            if m.distance < 0.75 * n.distance or m.distance < 100:
                good_matches.append([m])
                # Store the matching keypoints in a tuple in a list
                self.matching_keypoints.append((self.kp[m.trainIdx], self.kp_query[m.queryIdx],
                                                self.img_size_list[m.trainIdx], self.img_centroid_list[m.trainIdx],
                                                self.image_query_size))
        # Make sure size and scale give similar results
        test_size(self.matching_keypoints)
        print("Number of good matches: ", len(self.matching_keypoints))
        return self.matching_keypoints

    def apply_hough_transform(self, min_votes=3, paper=True):
        # cv2.drawMatchesKnn expects list of lists as matches.
        # img = cv2.drawKeypoints(rgb_query, queryImage_kp, None, flags=2)
        # Apply hough transform
        angle_factor = 20
        scale_factor = 2
        pos_factor = 16
        # Perform hough transform
        pose_bins = perform_hough_transform(self.matching_keypoints, angle_factor, scale_factor, pos_factor, paper)

        # Get most voted
        best_pose_bin = PoseBin()  # The best voted bin so far
        best_pose_bin.votes = min_votes  # set its minimum vote to 3
        dup_bins = []
        for key in pose_bins:
            if pose_bins.get(key).votes >= min_votes:  # all valid bins contain more than or equal to 3 votes
                self.valid_bins.append(pose_bins.get(key))
            if pose_bins.get(key).votes > best_pose_bin.votes:  # find the most voted for pose bin
                best_pose_bin = pose_bins.get(key)
                dup_bins = [pose_bins.get(key)]
            # keep track of other bins with similar number of votes
            elif pose_bins.get(key).votes == best_pose_bin.votes:
                dup_bins.append(pose_bins.get(key))

        print("Number of duplicate votes: ", len(dup_bins))

        return dup_bins, best_pose_bin.votes
        print("main done")

    def apply_affine_parameters(self):
        # Applying Affine parameters
        pos_factor = 32
        update = True
        num_updates = 0
        while update:  # While we are eliminating outliers keep applying affine parameters
            update = False
            max_vote = 3
            best_pose_bin = PoseBin()
            dup_bins = []
            remaining_bins = []
            for pose_bin in self.valid_bins:
                AffineParameters(pose_bin)  # Get Affine Parameters
                # Remove invalid keypoints
                _, change = remove_outliers(pose_bin, self.image_query_size, pos_factor * 4, pos_factor * 4)
                update = change or update  # if we changed the keypoints, set flag to true
                if pose_bin.votes >= 3:  # Get a list of remaining valid bins
                    remaining_bins.append(pose_bin)
                if pose_bin.votes > best_pose_bin.votes:  # Find which bins have the most votes
                    best_pose_bin = pose_bin
                    dup_bins = [pose_bin]
                elif pose_bin.votes == best_pose_bin.votes:  # store other bins with most votes
                    dup_bins.append(pose_bin)
            valid_bins = remaining_bins  # remaining bins are valid
            num_updates += 1
        # end while loop

        print(num_updates)
        print("affine parameters done")
        return dup_bins

    def plot_rects(self, dup_bins, single=true):
        if single:
            plot_multiple_rect(self.gray_query, dup_bins, self.ax)
        else:
            plot_single_rect_from_list(self.gray_query, dup_bins, self.ax)
        plt.show()


if __name__ == "__main__":
    sift = cv2.SIFT_create()
    main = Main()
    main.get_query_features("../Data_Set/Test dataset/Rotation/IMG_20211027_170134.jpg")
    main.run_matcher()
    dup_bins, max_votes = main.apply_hough_transform(3, False)
    main.plot_rects(dup_bins)
    # plt.show()

    # used_keypoints = []
    # for pose_bin in dup_bins:
    #     used_keypoints.extend([kp_pair[1] for kp_pair in pose_bin.keypoint_pairs])
    # remove_list = []
    # for kp in used_keypoints:
    #     for kp_info in main.matching_keypoints:
    #         if kp == kp_info[1]:
    #             main.matching_keypoints.remove(kp_info)
    #             break
    #
    # dup_bins = main.apply_hough_transform(max(max_votes/10, 3))
    # plt.show()
