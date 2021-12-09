import numpy as np

'''
Function to generate A matrix from model points
x and y are model points vectors
A is calculated based on the number of points we have in each bins
Each point contributes to 2 rows; one for x and the other for y
'''


def Gen_A(x, y, votes):
    A = []

    for k in range(0, votes):
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


def Gen_b(b_x, b_y, votes):
    b = []
    for k in range(0, votes):
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


'''
Algorithm

We will work with clusters formed in HT

Step 1: Import x and y points of query image and model image
Step 2: Calculate affine parameters
Step 3: Repeat the process for all the clusters

'''
def AffineParameters(posebin):
    x_vec = []
    y_vec = []
    b_x = []
    b_y = []

    for i in range(posebin.votes):
        x_vec.append(posebin.keypoint_pairs[i][0].pt[0])
        y_vec.append(posebin.keypoint_pairs[i][0].pt[1])
        b_x.append(posebin.keypoint_pairs[i][1].pt[0])
        b_y.append(posebin.keypoint_pairs[i][1].pt[1])

    A = Gen_A(x_vec, y_vec, posebin.votes)
    # b = [2.3, 4.3]
    b = Gen_b(b_x, b_y, posebin.votes)

    if A:
        x = Calc_x(A, b)
        [m1, m2, m3, m4, tx, ty] = Ext_Params(x)

        # After extracting Affine Parameters, we need to remove Outliers and repeat the process
        posebin.affine_parameters = [m1, m2, m3, m4, tx, ty]
        [m1, m2, m3, m4, tx, ty]
    else:
        [0, 0, 0, 0, 0, 0]


def remove_outliers(posebin, image_query_size, x_factor=8, y_factor=8):
    # Threshold values
    # x_ref = 132.8125 / 256
    # y_ref = 141.8125 / 256
    x_ref = image_query_size[0] * posebin.pose[3] / x_factor
    y_ref = image_query_size[1] * posebin.pose[3] / y_factor

    # Get the number of keypoint pairs in the bin
    num_of_kpp = len(posebin.keypoint_pairs)

    # Get AP for the bin
    M = []
    r1 = [posebin.affine_parameters[0], posebin.affine_parameters[1]]
    r2 = [posebin.affine_parameters[2], posebin.affine_parameters[3]]
    M.append(r1)
    M.append(r2)
    T = [posebin.affine_parameters[4], posebin.affine_parameters[5]]
    to_pop = []
    for i in range(0, num_of_kpp):

        # Get model (x, y) for the keypoint pair
        x_model = posebin.keypoint_pairs[i][0].pt[0]
        y_model = posebin.keypoint_pairs[i][0].pt[1]
        X = [x_model, y_model]

        # Get image (u, v) for the keypoint pair
        u_image = posebin.keypoint_pairs[i][1].pt[0]
        v_image = posebin.keypoint_pairs[i][1].pt[1]

        # Get image (u_prime, v_prime) from the AP calculations
        U1 = np.matmul(M, X)
        U = U1 + T
        U = U.tolist()

        u_AP = U[0]
        v_AP = U[1]

        # Decision
        if abs(u_AP - u_image) > x_ref or abs(v_AP - v_image) > y_ref:
            to_pop.append(i)
    update = len(to_pop) > 0
    posebin.keypoint_pairs = [v for i, v in enumerate(posebin.keypoint_pairs) if
                                          i not in frozenset(to_pop)]
    posebin.votes = len(posebin.keypoint_pairs)
    return posebin, update
