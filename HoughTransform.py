## Old version of hough transform

import math
from SiftHelperFunctions import *
from PoseBin import *
# from main import *

def perform_hough_transform(matching_keypoints, image_query, bin_x=15, bin_y=15, bin_theta=15, bin_sigma=15):
    hough_dict = {}

    img_height = image_query.shape[0]
    img_width = image_query.shape[1]

    for index, element in enumerate(matching_keypoints):
        kpM = element[0]
        kpQ = element[1]
        m_size = element[2]
        m_centroid = element[3]
        q_x = kpQ.pt[0]
        q_y = kpQ.pt[1]
        q_octave, q_layer, q_scale = unpack_sift_octave(kpQ)
        q_theta = kpQ.angle
        m_x = kpM.pt[0]
        m_y = kpM.pt[1]
        m_octave, m_layer, m_scale = unpack_sift_octave(kpM)
        m_theta = kpM.angle
        # print(q_theta, m_theta)
        #translation and scaling
        scale = m_scale / q_scale                           ## scale the model keypoints to query keypoint
        # scale = kpM.size/kpQ.size
        translated_x = (m_centroid[0] - m_x) * (scale)
        translated_y = (m_centroid[1] - m_y) * (scale)
        #rotation
        alpha = math.radians(q_theta - m_theta)
        rotated_x = math.cos(alpha) * translated_x - math.sin(alpha) * translated_y
        rotated_y = math.sin(alpha) * translated_x + math.cos(alpha) * translated_y
        #translate back to estimate the centroid of the query image
        q_centroid = (rotated_x + q_x, rotated_y + q_y)
        ##calculate x and y index for the hough transform bin
        i_x = int((q_centroid[0] * bin_x) / img_width)     
        i_x = max(0, i_x - 1)                                ## making sure the index does not go out of range
        i_x = min(i_x, bin_x - 1)                            ## making sure the index does not go out of range
        i_y = int((q_centroid[1] * bin_y) / img_height)  
        i_y = max(0, i_y- 1)                                 ## making sure the index does not go out of range
        i_y = min(i_y, bin_y - 1)                           ## making sure the index does not go out of range
        ##normalize theta from [-pi, pi] to [0, 2*pi] and calculate index
        # i_theta_prime = (alpha + math.pi) * bin_theta / (math.pi * 2)
        alpha = (alpha + 2*math.pi) % (2*math.pi)
        i_theta_prime = alpha * bin_theta / (2*math.pi)
        ##To allow for the rotations by -pi or pi to be close together
        i_theta = int(i_theta_prime % bin_theta)
        ##determine the scale index
        n_oct = 4                                           ##Assuming number of octaves used by the opencv sift_create() as 4.
        i_sigma = int((math.log(scale, 2) / (2 * (n_oct- 1) + 0.5) * bin_sigma))
        i_sigma = max(0, i_sigma)                           ## making sure the index does not go out of range
        i_sigma = min(i_sigma, bin_sigma - 1)               ## making sure the index does not go out of range
        for w in range(2):
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        a = i_x + w
                        b = i_y + x
                        c = i_theta + y
                        d = i_sigma + z
                        pose = (a, b, c, d)
                        if (a < bin_x and b < bin_y and c < bin_theta and d < bin_sigma):
                            try:
                                
                                hough_dict[pose].update_posebin(q_centroid, alpha, scale, m_size, (kpM, kpQ))
                            except KeyError:
                                mean = q_centroid, alpha, scale
                                hough_dict[pose] = PoseBin(pose, m_size, 1, [(kpM, kpQ)], mean)
    return hough_dict