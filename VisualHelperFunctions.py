from matplotlib import pyplot as plt
from matplotlib import patches as patches
from PoseBin import *
import cv2


# VISUALIZATION ###############################################################
def plot_rect(gray_img, pose_bin=PoseBin()):
    fig, ax = plt.subplots()
    img = cv2.drawKeypoints(gray_img, [x[1] for x in pose_bin.keypoint_pairs], None, None, flags=4)
    plt.imshow(img)
    pose = pose_bin.pose
    # add box to image
    IMG_WIDTH = pose_bin.img_size[0]
    IMG_HEIGHT = pose_bin.img_size[1]
    x_shift = -IMG_WIDTH * pose[3] / 2
    y_shift = -IMG_HEIGHT * pose[3] / 2

    # Determining the top left corner of the triangle with rotation
    rect_left_corner = (pose[0] + np.cos(np.deg2rad(pose[2])) * x_shift - np.sin(np.deg2rad(pose[2])) * y_shift,
                        pose[1] + np.sin(np.deg2rad(pose[2])) * x_shift + np.cos(np.deg2rad(pose[2])) * y_shift)

    rect = patches.Rectangle(rect_left_corner,
                             IMG_WIDTH * pose[3], IMG_HEIGHT * pose[3], pose[2],
                             linewidth=4, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    return fig, ax
