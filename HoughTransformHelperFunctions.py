import math
from SiftHelperFunctions import *

def estimate_object_pose(data):
    kpM = data[0] # Model Keypoint
    kpQ = data[1] # Query Keypoint
    # m_size = data[2] # Model image size
    m_centroid = data[3] # Model centroid
    # Get the x and y coordinates of the query keypoint
    q_x = kpQ.pt[0]
    q_y = kpQ.pt[1]
    # Get the scale of query keypoint
    q_octave, q_layer, q_scale = unpack_sift_octave(kpQ)
    q_theta = kpQ.angle # Get the orientation of query keypoint (range[0,360])
    # Get the x and y coordinates of the model keypoint
    m_x = kpM.pt[0]
    m_y = kpM.pt[1]
    # Get the scale of model keypoint
    m_octave, m_layer, m_scale = unpack_sift_octave(kpM)
    m_theta = kpM.angle # Get the orientation of model keypoint (range[0,360])

    scale_factor = m_scale / q_scale   # scale factor for scaling the model keypoints to query keypoint
    # scale = kpM.size/kpQ.size
    # Obtain direction vector from model keypoint to model centroid and scale it to query keypoint image size
    translated_x = (m_centroid[0] - m_x) * (scale_factor)
    translated_y = (m_centroid[1] - m_y) * (scale_factor)
    # rotate the translation vector to orient it as per query keypoint
    alpha = math.radians(q_theta - m_theta)
    # normalize theta from [-2pi, 2pi] to [0, 2*pi] and calculate index
    alpha = (alpha + 2*math.pi) % (2*math.pi)
    rotated_x = math.cos(alpha) * translated_x - math.sin(alpha) * translated_y
    rotated_y = math.sin(alpha) * translated_x + math.cos(alpha) * translated_y
    # translate back to estimate the centroid in query image
    x = rotated_x + q_x
    y = rotated_y + q_y
    object_pose = (x, y, alpha, scale_factor)
    return object_pose

def calculate_bin_index(object_pose, bins, query_image_shape):
    # Query image height and width
    img_height = query_image_shape[0]
    img_width = query_image_shape[1]
    x = object_pose[0]
    y = object_pose[1]
    theta = object_pose[2]
    scale_factor = object_pose[3]

    # calculate x bin index
    i_x = int((x * bins) / img_width)  
    # make sure the index is within range   
    i_x = max(0, i_x - 1)                          
    i_x = min(i_x, bins - 1)      

    # calculate y bin index                   
    i_y = int((y * bins) / img_height)  
    # make sure the index is within range
    i_y = max(0, i_y- 1)                                
    i_y = min(i_y, bins - 1)   

   
    i_theta_prime = theta * bins / (2*math.pi)
    # To allow for the rotations by 0 or 2pi to be close together
    i_theta = int(i_theta_prime % bins)

    # determine the scale index
    n_oct = 4  # Assuming number of octaves used by the opencv sift_create() as 4.
    i_sigma = int((math.log(scale_factor, 2) / (2 * (n_oct- 1) + 0.5) * bins))
    # make sure the index is within range
    i_sigma = max(0, i_sigma)                           
    i_sigma = min(i_sigma, bins - 1)

    return i_x, i_y, i_theta, i_sigma