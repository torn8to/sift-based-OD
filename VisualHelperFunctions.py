from matplotlib import pyplot as plt
from matplotlib import patches as patches
from PoseBin import *
import cv2
from SiftHelperFunctions import average_poses


# VISUALIZATION ###############################################################
def plot_rect(gray_img, pose_bin, ax, color='r', show_kp=True):
    if show_kp:
        show_keypoints(gray_img, pose_bin.keypoint_pairs, ax)
    else:
        plt.imshow(gray_img, cmap='gray')
    x_pose = pose_bin.pose[0]
    y_pose = pose_bin.pose[1]
    ori = pose_bin.pose[2]
    scale = pose_bin.pose[3]
    # add box to image
    IMG_WIDTH = pose_bin.img_size[0]
    IMG_HEIGHT = pose_bin.img_size[1]
    x_shift = -IMG_WIDTH * scale / 2
    y_shift = -IMG_HEIGHT * scale / 2

    # Determining the top left corner of the triangle with rotation
    rect_left_corner = (x_pose + np.cos(np.deg2rad(ori)) * x_shift - np.sin(np.deg2rad(ori)) * y_shift,
                        y_pose + np.sin(np.deg2rad(ori)) * x_shift + np.cos(np.deg2rad(ori)) * y_shift)

    rect = patches.Rectangle(rect_left_corner,
                             IMG_WIDTH * scale, IMG_HEIGHT * scale, ori,
                             linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    return ax


def plot_multiple_rect(gray_img, dup_bins, ax, show_kp=True):
    ax.clear()
    color_count = 0
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'w']
    for pose_bin in dup_bins:
        ax = plot_rect(gray_img, pose_bin, ax, colors[color_count % len(colors)], show_kp)
        color_count += 1
    return ax


def plot_single_rect_from_list(gray_img, dup_bins, ax, show_kp=True):
    pose_ideal = dup_bins[0]
    pose_ideal.pose = average_poses([v.pose for v in dup_bins], show_kp)
    plot_rect(gray_img, pose_ideal, ax)


def show_keypoints(rgb_query, keypoint_pairs, ax):
    img = cv2.drawKeypoints(rgb_query, [x[1] for x in keypoint_pairs], None, flags=4)
    plt.imshow(rgb_query, cmap='gray')

    offset = 30
    for kpM, kpQ in keypoint_pairs:
        x = kpQ.pt[0] - offset
        y = kpQ.pt[1] - offset
        rect_left_corner = x, y
        rect = patches.Rectangle(rect_left_corner,
                            2*offset, 2*offset, 0,
                            linewidth=0.5, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    return ax