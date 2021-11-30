import numpy as np


class PoseBin:

    def __init__(self, pose = (0, 0, 0, 0), img_size=(100, 100), votes=0, keypoint_pairs=[]):
        self.pose = pose  # (x, y, theta, scale)
        self.img_size = img_size  # (width, height)
        self.votes = votes
        self.keypoint_pairs = keypoint_pairs

    def add_keypoint_pair(self, pair):
        self.keypoint_pairs.append(pair)

    def update_img_size(self, img_size):
        old_width = self.img_size[0]
        old_height = self.img_size[1]

        new_width = (old_width * self.votes + img_size[0]) / (self.votes + 1)
        new_height = (old_height * self.votes + img_size[1]) / (self.votes + 1)
        self.img_size = (new_width, new_height)

    def add_vote(self, new_votes=1):
        self.votes += new_votes

    def get_pts(self):
        # the x should have its own vector and y its own vector
        training_x = np.array([])
        training_y = np.array([])
        model_x = np.array([])
        model_y = np.array([])
        for kp_pair in self.keypoint_pairs:
            training_x = np.append(training_x, kp_pair[1].pt[0])
            training_y = np.append(training_y, kp_pair[1].pt[1])
            model_x = np.append(model_x, kp_pair[0].pt[0])
            model_y = np.append(model_y, kp_pair[0].pt[1])
        return training_x, training_y, model_x, model_y

    def remove_keypoint_pair(self, pair):
        self.keypoint_pairs.remove(pair)

    def is_same_pose(self, pose):
        return pose == self.pose

    def __eq__(self, other):
        if isinstance(other, PoseBin):
            return other.pose == self.pose
        else:
            return False

    def __repr__(self):
        return "[" + str(self.pose) + ", " + str(self.votes) + ", " + str(self.img_size) + "]"

    def __str__(self):
        return "Pose: " + str(self.pose)
