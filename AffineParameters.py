import numpy as np
from main import *


'''
Function to generate A matrix from model points
x and y are model points vectors
A is calculated based on the number of points we have in each bins
Each point contributes to 2 rows; one for x and the other for y
'''
def Gen_A(x, y, bin_number):
    A = []
  
    for k in range(0, valid_bins[bin_number].votes):
        row_1 = [x[k], y[k], 0, 0, 1, 0]
        row_2 = [0, 0, x[k], y[k], 0, 1]
        A.append(row_1)
        A.append(row_2)

    return A

'''
Function to generate b matrix from image points
b_x and b_y are model points vectors
b is calculated based on the number of points we have in each bins
Each point contributes to 2 rows; one for x and the other for y
'''
def Gen_b(b_x, b_y, bin_number):
    b = []
    for k in range(0, valid_bins[bin_number].votes):
        row_1 = b_x[k]
        row_2 = b_y[k]
        b.append(row_1)
        b.append(row_2)

    return b




'''
Function to calculate the least squares 
A is the vector with model points
b is the vector with image points
'''
def Calc_x(A, b):
    A_Transpose = np.transpose(A).tolist()
    mul = np.matmul(A_Transpose, A)
    mul = np.linalg.pinv(mul)
    x = np.matmul(A_Transpose, b)
    x = np.matmul(mul, x)
    return x


'''
A is calculated based on the number of points we have in each bins
Each points contributes to 2 rows; one for x and  the other for y
'''

'''
Function to extract each parameter
After calulating x, we need to extract each parameter
'''
def Ext_Params(x):
    m1 = x[0]
    m2 = x[1]
    m3 = x[2]
    m4 = x[3]
    tx = x[4]
    ty = x[5]
    return m1, m2, m3, m4, tx, ty


def AffineParameters(bin_number):
    
    x_vec = []
    y_vec = []
    b_x = []
    b_y = []

    for i in range(0, valid_bins[bin_number].votes):
        x_vec.append(valid_bins[bin_number].keypoint_pairs[i][0].pt[0])
        y_vec.append(valid_bins[bin_number].keypoint_pairs[i][0].pt[1])
        b_x.append(valid_bins[bin_number].keypoint_pairs[i][1].pt[0])
        b_y.append(valid_bins[bin_number].keypoint_pairs[i][1].pt[1])


    A = Gen_A(x_vec, y_vec, bin_number)
    #b = [2.3, 4.3]
    b = Gen_b(b_x, b_y, bin_number)

    if A:
        x = Calc_x(A, b)
        [m1, m2, m3, m4, tx, ty] = Ext_Params(x)

        # After extracting Affine Parameters, we need to remove Outliers and repeat the process

        
        return m1, m2, m3, m4, tx, ty       

    else: 

        return 0, 0, 0, 0, 0, 0
    '''
    Algorithm

    We will work with clusters formed in HT

    Step 1: Import x and y points of query image and model image
    Step 2: Calculate affine parameters
    Step 3: Repeat the process for all the clusters

    '''

#print("done")
