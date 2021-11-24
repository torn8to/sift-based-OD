from matplotlib import pyplot as plt
from matplotlib import patches as patches
import pickle
from cv2 import sort
from SiftHelperFunctions import *
from PoseBin import *
import sys


def perform_hough_transform(matching_keypoints, angle_breakpoint=10.0, scale_breakpoint=2.0, pos_factor=32.0):
    # Generate Pose guess of keypoints
    pose_bins = {}
    for kpM, kpQ, img_size, img_centroid in matching_keypoints:

        octaveM, layerM, scaleM = unpack_sift_octave(kpM)  # unpack octave information for model keypoint
        octaveQ, layerQ, scaleQ = unpack_sift_octave(kpQ)  # unpack octave information for query keypoint

        # Changed from LOWE
        x_pos_breakpoint = img_size[0] * scaleQ / pos_factor  # determine x axis bucket size in pixels
        y_pos_breakpoint = img_size[1] * scaleQ / pos_factor  # determine y axis bucket size in pixels

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

        possible_scale = [scale_breakpoint ** np.floor(np.log(pose_estimate[3]) / np.log(scale_breakpoint)),
                          scale_breakpoint ** np.ceil(np.log(pose_estimate[3]) / np.log(scale_breakpoint))]
        if possible_scale[0] == possible_scale[1]:
            # Sometimes the scale adds the same value twice leading to duplicates in the pose bins
            possible_scale.remove(possible_scale[1])

        for i in range(2):
            for j in range(2):
                for theta in range(2):
                    for s in range(len(possible_scale)):
                        pose = (possible_x_pos[i], possible_y_pos[j], possible_orientation[theta],
                                possible_scale[s])
                        try:
                            """
                            we update the vote
                            """
                            # Update the vote
                            pose_bins[pose].add_vote()
                            # update the average object size
                            pose_bins[pose].update_img_size(img_size)
                            # update the keypoint list
                            pose_bins[pose].add_keypoint_pair((kpM, kpQ))

                        except KeyError:
                            pose_bins[pose] = PoseBin(pose, img_size, 1, [(kpM, kpQ)])

    return pose_bins
