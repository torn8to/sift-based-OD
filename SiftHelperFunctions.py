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
    centroid = []
    for point in temp_kp:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2], response=point[3],
                            octave=point[4], class_id=point[5])
        kp.append(temp)
        centroid.append(point[6])
    return kp, centroid


def make_temp_kp(kp):
    temp_kp = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        temp_kp.append(temp)
    return temp_kp
