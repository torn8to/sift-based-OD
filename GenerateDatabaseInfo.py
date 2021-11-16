import cv2
from matplotlib import pyplot as plt
import os
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# calculates the centroid for the keypoints generated from one training image
def calculate_centroid(kp):
    x_centroid = 0
    y_centroid = 0
    total = len(kp)
    for point in kp:
        x_centroid = x_centroid + point.pt[0]
        y_centroid = y_centroid + point.pt[1]
    x_centroid = x_centroid / total
    y_centroid = y_centroid / total
    return x_centroid, y_centroid

def make_temp_kp(kp, shape):
    temp_kp = []
    x_centroid, y_centroid = calculate_centroid(kp)
    centroid = x_centroid, y_centroid
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id, centroid, shape)
        temp_kp.append(temp)             #edited temp_kp to include the centroid each point in a keypoint list
                                         #for each training image
    return temp_kp


data = []

sift = cv2.SIFT_create()
path1 = '/home/prajwal/Desktop/cv_group_project/D1/'
listing = os.listdir(path1)
# listing.remove('.git')
for file in listing:
    img = cv2.imread(path1 + file)
    shape = img.shape
    # image comparison
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # need to do some more processing here

    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(gray_img, None)
    temp_kp = make_temp_kp(kp, shape) ##keeping track of image height and width for each keypoint

    data.append([temp_kp, des, path1 + file])
save_object(data, 'training_data.pkl')
