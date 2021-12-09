import cv2

# calculates the centroid for the keypoints generated from one training image
def get_centroid(kp):
    x_centroid = 0
    y_centroid = 0
    total = len(kp)
    for point in kp:
        x_centroid = x_centroid + point.pt[0]
        y_centroid = y_centroid + point.pt[1]
    x_centroid = x_centroid / total
    y_centroid = y_centroid / total
    centroid = x_centroid, y_centroid
    return centroid

def make_temp_kp(kp):
    temp_kp = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        temp_kp.append(temp)             
    return temp_kp


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
    shape = []
    for point in temp_kp:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        kp.append(temp)
    return kp


