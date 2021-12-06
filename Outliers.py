from Appended_AP import *
import numpy as np


def remove_outliers(bin_number):
    
    # Threshold values
    x_ref = 132.8125 / 256
    y_ref = 141.8125 / 256

    # Get the number of keypoint pairs in the bin
    num_of_kpp = len(valid_bins[bin_number].keypoint_pairs)

    # Get AP for the bin
    M = []
    r1 = [appended_affine_parameters[bin_number][0], appended_affine_parameters[bin_number][1]]
    r2 = [appended_affine_parameters[bin_number][2], appended_affine_parameters[bin_number][3]]
    M.append(r1)
    M.append(r2)
    T = [appended_affine_parameters[bin_number][4], appended_affine_parameters[bin_number][5]]
    to_pop = []
    for i in range(0, num_of_kpp):

        # Get model (x, y) for the keypoint pair
        x_model = valid_bins[bin_number].keypoint_pairs[i][0].pt[0]
        y_model = valid_bins[bin_number].keypoint_pairs[i][0].pt[1]
        X = [x_model, y_model]

        # Get image (u, v) for the keypoint pair
        u_image = valid_bins[bin_number].keypoint_pairs[i][1].pt[0]
        v_image = valid_bins[bin_number].keypoint_pairs[i][1].pt[1]

        # Get image (u_prime, v_prime) from the AP calculations
        U1 = np.matmul(M, X)
        U = U1 + T
        U = U.tolist()
        
        u_AP = U[0]
        v_AP = U[1]

        
        # Decision
        if abs(u_AP - u_image) > x_ref or abs(v_AP - v_image) > y_ref:
            to_pop.append(i)

    valid_bins[bin_number].keypoint_pairs = [v for i,v in enumerate(valid_bins[bin_number].keypoint_pairs) if i not in frozenset(to_pop)]
    valid_bins[bin_number].votes = len(valid_bins[bin_number].keypoint_pairs)







    
