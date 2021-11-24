class PoseBin:

    def __init__(self, pose, img_size=(100, 100), votes=0, keypoint_pairs=[]):
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
        # TODO get the x and y points for each keypoint pair
        # the x should have its own vector and y its own vector
        pass

    def remove_keypoint_pair(self, pair):
        # TODO remove the keypoint pair and subtract one from the vote
        pass

