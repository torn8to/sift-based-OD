import numpy as np


class PoseBin:

    def __init__(self, pose, img_size=(100, 100), votes=0, keypoint_pairs=[], mean= (0, 0, 0, 0)):
        self.pose = pose  # (x, y, theta, scale)
        self.img_size = img_size  # (width, height)
        self.votes = votes
        self.keypoint_pairs = keypoint_pairs
        self.centroid = mean[0], mean[1]
        self.angle = mean[2]
        self.scale = mean[3]
        self.affine_parameters = []

    def add_keypoint_pair(self, pair):
        self.keypoint_pairs.append(pair)

    def update_centroid(self, center):
        old_centroid_x = self.centroid[0]
        old_centroid_y = self.centroid[1]

        new_centroid_x = (old_centroid_x * self.votes + center[0]) / (self.votes + 1)
        new_centroid_y = (old_centroid_y * self.votes + center[1]) / (self.votes + 1)
        self.centroid = (new_centroid_x, new_centroid_y)

    def update_angle(self, alpha):
        old_alpha = self.angle
        new_alpha = (old_alpha * self.votes + alpha) / (self.votes + 1)
        self.angle = new_alpha
    
    def update_scale(self, scale):
        old_scale = self.scale
        new_scale = (old_scale * self.votes + scale) / (self.votes + 1)
        self.scale = new_scale

    def update_img_size(self, img_size):
        old_width = self.img_size[0]
        old_height = self.img_size[1]

        new_width = (old_width * self.votes + img_size[0]) / (self.votes + 1)
        new_height = (old_height * self.votes + img_size[1]) / (self.votes + 1)
        self.img_size = (new_width, new_height)

    def update_posebin(self, object_pose, img_size, keypoint_pair):
        self.update_centroid((object_pose[0], object_pose[1]))
        self.update_angle(object_pose[2])
        self.update_scale(object_pose[3])
        self.update_img_size(img_size)
        self.add_keypoint_pair(keypoint_pair)
        self.add_vote()

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