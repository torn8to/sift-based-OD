import cv2
import numpy as np


def unpack_sift_octave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)
    return (octave, layer, scale)


def make_kp(temp_kp):
    kp = []
    for point in temp_kp:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                            _response=point[3], _octave=point[4], _class_id=point[5])
        kp.append(temp)
    return kp


def make_temp_kp(kp):
    temp_kp = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        temp_kp.append(temp)
    return temp_kp


def normalize_angle(angle):
    angle = np.fmod(angle, 360)
    angle = np.fmod(angle + 360, 360)
    if (angle > 180):
        angle -= 360
    return angle


def get_centroid(kp):
    centroid = (0, 0)
    count = 0
    max_octave = 0
    for keypoint in kp:
        octave, _, _ = unpack_sift_octave(keypoint)
        max_octave = max(max_octave, octave)
        x, y = keypoint.pt

        # Just averaging x and y positions
        new_x = (centroid[0] * count + x) / (count + 1)
        new_y = (centroid[1] * count + y) / (count + 1)
        centroid = (new_x, new_y)
        count += 1
    return centroid