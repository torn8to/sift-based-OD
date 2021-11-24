class PoseBin:

    def __init__(self, pose, img_size=(100, 100), votes=1, keypoint_pairs=[]):
        self.pose = pose
        self.img_size = img_size
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



