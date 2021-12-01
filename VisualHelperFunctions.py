from matplotlib import pyplot as plt
from matplotlib import patches as patches
# from PoseBin import *
import cv2
import math

def plot_rect(rgb_query, final_pose, keypoint_pairs):
    fig, ax = plt.subplots()
    img = cv2.drawKeypoints(rgb_query, [x[1] for x in keypoint_pairs], None, flags=4)
    plt.imshow(img)

    for pose in final_pose:
        print("Object pose is:")
        print(pose[0], math.degrees(pose[1]), pose[2], pose[3])
        x_prime = - pose[3][0] *(pose[2]) / 2
        y_prime = - pose[3][1] * (pose[2]) / 2
        theta = pose[1]
        x_dash = math.cos(theta) * x_prime - math.sin(theta) * y_prime
        y_dash = math.sin(theta) * x_prime + math.cos(theta) * y_prime
        x = pose[0][0] + x_dash
        y = pose[0][1] + y_dash
        rect_left_corner = x, y
        rect = patches.Rectangle(rect_left_corner,
                            pose[3][0] * pose[2], pose[3][1] * pose[2], math.degrees(pose[1]),
                            linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    return fig, ax