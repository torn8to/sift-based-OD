import cv2
from matplotlib import pyplot as plt
import os
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def make_temp_kp(kp):
    temp_kp = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        temp_kp.append(temp)
    return temp_kp


data = []

sift = cv2.SIFT_create()
path1 = '../../../../../Multi_Modal_Image_Classifier/Multi_Modal_Image_Classifier/'
listing = os.listdir(path1)
for file in listing:
    img = cv2.imread(path1 + file)

    # image comparison
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # need to do some more processing here

    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(gray_img, None)

    temp_kp = make_temp_kp(kp)
    data.append([temp_kp, des, path1 + file])

save_object(data, 'training_data.pkl')
