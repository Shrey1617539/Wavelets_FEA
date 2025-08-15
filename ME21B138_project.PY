import numpy as np
import matplotlib.pyplot as plt

def f_function(x):
    return ((np.pi**2)*np.sin(np.pi*x)) + np.sin(np.pi*x)

def epsilon_function(x):
    return 1

def gamma_function(x):
    return 1

# giving gaussina points details
def gauss_pt_details(no_of_gauss_pt):
    if no_of_gauss_pt==1:
        gauss_pt = np.array([0])
        weights = np.array([2])
        return gauss_pt, weights
    
    if no_of_gauss_pt == 2:
        gauss_pt = np.array([-1*np.sqrt(1/3), np.sqrt(1/3)])
        weights = np.array([1,1])
        return gauss_pt, weights
    
    if no_of_gauss_pt == 3:
        gauss_pt = np.array([-1*np.sqrt(3/5), 0, np.sqrt(3/5)])
        weights = np.array([5/9, 8/9, 5/9])
        return gauss_pt, weights

# function to generate and assemble the corase grid matrices 
def matrix_cal_1D(mesh, coordinates, no_of_gauss_pts = 3):

    K = np.zeros((coordinates.shape[0], coordinates.shape[0]))
    M = np.zeros((coordinates.shape[0], coordinates.shape[0]))
    F = np.zeros((coordinates.shape[0],1))

    for ele in range(mesh.shape[0]):
        con = mesh[ele:ele+1]
        coord = coordinates[con]

        gauss_pts_KM, weights_KM = gauss_pt_details(no_of_gauss_pts)

        k = np.zeros((2,2))
        m = np.zeros((2,2))
        f = np.zeros((2,1))
        
        for j in range(no_of_gauss_pts):
            gauss_pt = gauss_pts_KM[j]
            weight = weights_KM[j]

            N = np.array([[(1-gauss_pt)/2, (1+gauss_pt)/2]])
            dN = np.array([[-1/2, 1/2]])
            J = dN.dot(coord.T)
            dphi = np.linalg.inv(J).dot(dN)
            X = N.dot(coord.T)

            f += (N.T)*f_function(X)*np.linalg.det(J)*weight
            k += (dphi.T.dot(dphi))*epsilon_function(X)*np.linalg.det(J)*weight
            m += (N.T.dot(N))*gamma_function(X)*np.linalg.det(J)*weight
        Zeros = np.zeros((con.shape[0],)).astype(int)
        
        F[np.ix_(con[0],Zeros)] +=  f
        K[np.ix_(con[0], con[0])] += k
        M[np.ix_(con[0], con[0])] += m

    F = np.squeeze(F)

    return K, M, F

def elemental_matrix_cal_1D(no_of_additional_scales, coord, no_of_gauss_pts = 3):
    # total number of nodes
    total_nodes = 2**(no_of_additional_scales) + 1

    K = np.zeros((total_nodes, total_nodes))
    M = np.zeros((total_nodes, total_nodes))
    F = np.zeros((total_nodes,))
    coordinates = np.zeros((total_nodes-2,))

    gauss_pts, weights = gauss_pt_details(no_of_gauss_pts)

    dN = np.array([[-1/2, 1/2]])
    J = dN.dot(coord.T)

    # Calculating the corase grid for given element
    for pt_id in range(weights.shape[0]):
        N = np.array([[(1-gauss_pts[pt_id])/2, (1+gauss_pts[pt_id])/2]])
        dphi = np.linalg.inv(J).dot(dN)
        X = N.dot(coord.T)
        F[:2] += np.squeeze((N.T)*f_function(X)*np.linalg.det(J)*weights[pt_id])
        K[:2, :2] += (dphi.T.dot(dphi))*epsilon_function(X)*np.linalg.det(J)*weights[pt_id]
        M[:2, :2] += (N.T.dot(N))*gamma_function(X)*np.linalg.det(J)*weights[pt_id]
    

    for l1 in range(no_of_additional_scales):
        for j1 in range(2**(l1)):

            # Finding active region for first pseduo-wavelet shape function
            left_bou_1 = ((2**(1-l1))*j1)-1
            center_1 = ((2**(1-l1))*j1) + (2**(-l1)) - 1
            right_bou_1 = ((2**(1-l1))*j1) + (2**(1-l1)) - 1
            
            # finding the real coordinates of active region
            sub_coord_1_1 = (np.array([[(1-left_bou_1)/2, (1+left_bou_1)/2], [(1-center_1)/2, (1+center_1)/2]]).dot(coord.T)).T
            sub_coord_1_2 = (np.array([[(1-center_1)/2, (1+center_1)/2], [(1-right_bou_1)/2, (1+right_bou_1)/2]]).dot(coord.T)).T
            coordinates[(2**l1) + j1 -1] = sub_coord_1_1[0][1]

            # Finding matricies for interaction between first pseudo shape fucntion and corase grid elements with numerical integrations
            for pt_id in range(weights.shape[0]):
                N = np.array([[(1-gauss_pts[pt_id])/2, (1+gauss_pts[pt_id])/2]])
                dN = np.array([[-1/2, 1/2]])
                X1_1 = (N.dot(sub_coord_1_1.T))[0][0]
                X1_2 = (N.dot(sub_coord_1_2.T))[0][0]
                sub_J_1_1 = dN.dot(sub_coord_1_1.T)
                sub_J_1_2 = dN.dot(sub_coord_1_2.T)

                M[0][(2**l1) + j1 + 1] += ((X1_1 - coord[0][1])/(coord[0][0]-coord[0][1]))*((X1_1 - sub_coord_1_1[0][0])/(sub_coord_1_1[0][1] - sub_coord_1_1[0][0]))*gamma_function(X1_1)*np.linalg.det(sub_J_1_1)*weights[pt_id]
                M[0][(2**l1) + j1 + 1] += ((X1_2 - coord[0][1])/(coord[0][0]-coord[0][1]))*((X1_2 - sub_coord_1_2[0][1])/(sub_coord_1_2[0][0] - sub_coord_1_2[0][1]))*gamma_function(X1_2)*np.linalg.det(sub_J_1_2)*weights[pt_id]
                M[1][(2**l1) + j1 + 1] += ((X1_1 - coord[0][0])/(coord[0][1]-coord[0][0]))*((X1_1 - sub_coord_1_1[0][0])/(sub_coord_1_1[0][1] - sub_coord_1_1[0][0]))*gamma_function(X1_1)*np.linalg.det(sub_J_1_1)*weights[pt_id]
                M[1][(2**l1) + j1 + 1] += ((X1_2 - coord[0][0])/(coord[0][1]-coord[0][0]))*((X1_2 - sub_coord_1_2[0][1])/(sub_coord_1_2[0][0] - sub_coord_1_2[0][1]))*gamma_function(X1_2)*np.linalg.det(sub_J_1_2)*weights[pt_id]
                
                K[0][(2**l1) + j1 + 1] += (np.linalg.inv(J)[0][0]*dN[0][0])*(np.linalg.inv(sub_J_1_1)[0][0]*dN[0][1])*epsilon_function(X1_1)*np.linalg.det(sub_J_1_1)*weights[pt_id]
                K[0][(2**l1) + j1 + 1] += (np.linalg.inv(J)[0][0]*dN[0][0])*(np.linalg.inv(sub_J_1_2)[0][0]*dN[0][0])*epsilon_function(X1_2)*np.linalg.det(sub_J_1_2)*weights[pt_id] 
                K[1][(2**l1) + j1 + 1] += (np.linalg.inv(J)[0][0]*dN[0][1])*(np.linalg.inv(sub_J_1_1)[0][0]*dN[0][1])*epsilon_function(X1_1)*np.linalg.det(sub_J_1_1)*weights[pt_id]
                K[1][(2**l1) + j1 + 1] += (np.linalg.inv(J)[0][0]*dN[0][1])*(np.linalg.inv(sub_J_1_2)[0][0]*dN[0][0])*epsilon_function(X1_2)*np.linalg.det(sub_J_1_2)*weights[pt_id] 

                F[(2**l1) + j1 + 1] += ((X1_1 - sub_coord_1_1[0][0])/(sub_coord_1_1[0][1] - sub_coord_1_1[0][0]))*f_function(X1_1)*np.linalg.det(sub_J_1_1)*weights[pt_id]
                F[(2**l1) + j1 + 1] += ((X1_2 - sub_coord_1_2[0][1])/(sub_coord_1_2[0][0] - sub_coord_1_2[0][1]))*f_function(X1_2)*np.linalg.det(sub_J_1_2)*weights[pt_id]

            M[(2**l1) + j1 + 1][1] = M[1][(2**l1) + j1 + 1]
            M[(2**l1) + j1 + 1][0] = M[0][(2**l1) + j1 + 1] 
            K[(2**l1) + j1 + 1][1] = K[1][(2**l1) + j1 + 1]
            K[(2**l1) + j1 + 1][0] = K[0][(2**l1) + j1 + 1] 


            for l2 in range(l1, no_of_additional_scales):
                for j2 in range(2**(l2)):

                    # finding active region of 2nd pseudo-wavelet function
                    left_bou_2 = ((2**(1-l2))*j2)-1
                    center_2 = ((2**(1-l2))*j2) + (2**(-l2)) - 1
                    right_bou_2 = ((2**(1-l2))*j2) + (2**(1-l2)) - 1
                    
                    # finding if two wavelets are interacting with eachother or not
                    if (left_bou_1 <= left_bou_2) & (right_bou_1 >= right_bou_2):
                        dN = np.array([[-1/2, 1/2]])
                        sub_coord_2_1 = (np.array([[(1-left_bou_2)/2, (1+left_bou_2)/2], [(1-center_2)/2, (1+center_2)/2]]).dot(coord.T)).T
                        sub_coord_2_2 = (np.array([[(1-center_2)/2, (1+center_2)/2], [(1-right_bou_2)/2, (1+right_bou_2)/2]]).dot(coord.T)).T

                        # finding the matricies values by interacting with 1st and 2nd pseudo wavelet shape functions
                        for pt_id in range(weights.shape[0]):
                            N = np.array([[(1-gauss_pts[pt_id])/2, (1+gauss_pts[pt_id])/2]])
                            dN = np.array([[-1/2, 1/2]])
                            
                            X_1 = (N.dot(sub_coord_2_1.T))[0][0]
                            X_2 = (N.dot(sub_coord_2_2.T))[0][0]
                            
                            sub_J_2_1 = dN.dot(sub_coord_2_1.T)
                            sub_J_2_2 = dN.dot(sub_coord_2_2.T)
                            
                            if center_1 >= right_bou_2:
                                M[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += ((X_1 - sub_coord_1_1[0][0])/(sub_coord_1_1[0][1] - sub_coord_1_1[0][0]))*((X_1 - sub_coord_2_1[0][0])/(sub_coord_2_1[0][1] - sub_coord_2_1[0][0]))*gamma_function(X_1)*np.linalg.det(sub_J_2_1)*weights[pt_id]
                                M[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += ((X_2 - sub_coord_1_1[0][0])/(sub_coord_1_1[0][1] - sub_coord_1_1[0][0]))*((X_2 - sub_coord_2_2[0][1])/(sub_coord_2_2[0][0] - sub_coord_2_2[0][1]))*gamma_function(X_2)*np.linalg.det(sub_J_2_2)*weights[pt_id]
                                K[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += (np.linalg.inv(sub_J_1_1)[0][0]*dN[0][1])*(np.linalg.inv(sub_J_2_1)[0][0]*dN[0][1])*epsilon_function(X_1)*np.linalg.det(sub_J_2_1)*weights[pt_id]
                                K[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += (np.linalg.inv(sub_J_1_1)[0][0]*dN[0][1])*(np.linalg.inv(sub_J_2_2)[0][0]*dN[0][0])*epsilon_function(X_2)*np.linalg.det(sub_J_2_2)*weights[pt_id]

                            elif center_1 <= left_bou_2:
                                M[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += ((X_1 - sub_coord_1_2[0][1])/(sub_coord_1_2[0][0] - sub_coord_1_2[0][1]))*((X_1 - sub_coord_2_1[0][0])/(sub_coord_2_1[0][1] - sub_coord_2_1[0][0]))*gamma_function(X_1)*np.linalg.det(sub_J_2_1)*weights[pt_id]
                                M[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += ((X_2 - sub_coord_1_2[0][1])/(sub_coord_1_2[0][0] - sub_coord_1_2[0][1]))*((X_2 - sub_coord_2_2[0][1])/(sub_coord_2_2[0][0] - sub_coord_2_2[0][1]))*gamma_function(X_2)*np.linalg.det(sub_J_2_2)*weights[pt_id]
                                K[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += (np.linalg.inv(sub_J_1_2)[0][0]*dN[0][0])*(np.linalg.inv(sub_J_2_1)[0][0]*dN[0][1])*epsilon_function(X_1)*np.linalg.det(sub_J_2_1)*weights[pt_id]
                                K[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += (np.linalg.inv(sub_J_1_2)[0][0]*dN[0][0])*(np.linalg.inv(sub_J_2_2)[0][0]*dN[0][0])*epsilon_function(X_2)*np.linalg.det(sub_J_2_2)*weights[pt_id]

                            else:
                                M[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += ((X_1 - sub_coord_1_1[0][0])/(sub_coord_1_1[0][1] - sub_coord_1_1[0][0]))*((X_1 - sub_coord_2_1[0][0])/(sub_coord_2_1[0][1] - sub_coord_2_1[0][0]))*gamma_function(X_1)*np.linalg.det(sub_J_2_1)*weights[pt_id]
                                M[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += ((X_2 - sub_coord_1_2[0][1])/(sub_coord_1_2[0][0] - sub_coord_1_2[0][1]))*((X_2 - sub_coord_2_2[0][1])/(sub_coord_2_2[0][0] - sub_coord_2_2[0][1]))*gamma_function(X_2)*np.linalg.det(sub_J_2_2)*weights[pt_id]
                                K[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += (np.linalg.inv(sub_J_1_1)[0][0]*dN[0][1])*(np.linalg.inv(sub_J_2_1)[0][0]*dN[0][1])*epsilon_function(X_1)*np.linalg.det(sub_J_2_1)*weights[pt_id]
                                K[(2**l1) + j1 + 1][(2**l2) + j2 + 1] += (np.linalg.inv(sub_J_1_2)[0][0]*dN[0][0])*(np.linalg.inv(sub_J_2_2)[0][0]*dN[0][0])*epsilon_function(X_2)*np.linalg.det(sub_J_2_2)*weights[pt_id]


                            M[(2**l2) + j2 + 1][(2**l1) + j1 + 1] = M[(2**l1) + j1 + 1][(2**l2) + j2 + 1]
                            K[(2**l2) + j2 + 1][(2**l1) + j1 + 1] = K[(2**l1) + j1 + 1][(2**l2) + j2 + 1]

    return K, M, F, coordinates

def find_varible_values(mesh, coordinates, wavelet_level):
    # finding M, K, F matrix for corase grid
    K, M, F = matrix_cal_1D(mesh, coordinates, 3)
    U = np.zeros(F.shape)

    # applying appropriate boundary conditions
    U[1:-1] = np.linalg.inv((M[1:-1, 1:-1]) + (K[1:-1, 1:-1])).dot(F[1:-1])

    final_coordinate = coordinates
    final_U = U

    # solving for finer layers in elements
    for ele in range(mesh.shape[0]):
        con = mesh[ele:ele+1]
        coord = coordinates[con]
        U_end = U[con]

        # getting elemental matrices
        K_ele, M_ele, F_ele, coordinates_ele = elemental_matrix_cal_1D(wavelet_level, coord, 3)

        delta_u = np.zeros(F_ele.shape)
        U_ele = np.zeros(F_ele.shape)

        # solving for delta u
        U_ele[:2] = U_end[0]
        delta_u[:2] = U_end[0]
        delta_u[2:] = np.linalg.inv((M_ele[2:, 2:]) + (K_ele[2:, 2:])).dot(F_ele[2:]-(((M_ele[2:, :2]) + (K_ele[2:, :2])).dot(U_ele[:2])))
        
        # assembly to get final nodal values
        for l in range(wavelet_level):
            for j in range(2**l):
                curr = 2**l + j + 1
                l1 = 0
                while curr <= (2**wavelet_level) + 1:
                    if l1 == 0:
                        U_ele[curr] += delta_u[curr]
                        U_ele[curr] += ((2*j + 1)/2**(l+1))*delta_u[1]
                        U_ele[curr] += (1 - ((2*j + 1)/2**(l+1)))*delta_u[0]
                    else:
                        for u in range(2**(l1-1)):
                            U_ele[curr - u] += ((2*u + 1)/(2**(l1)))*delta_u[2**l + j + 1]
                            U_ele[curr - (2**l1) + 1 + u] += ((2*u + 1)/(2**(l1)))*delta_u[2**l + j + 1]
                    curr *= 2
                    l1 += 1
        gain = (np.linalg.norm(delta_u[-2**(wavelet_level-1)])/ np.linalg.norm(U_ele[-2**(wavelet_level-1)]))
        final_coordinate = np.concatenate((final_coordinate, coordinates_ele))
        final_U = np.concatenate((final_U, U_ele[2:]))

    sorted_indicies = np.argsort(final_coordinate)
    final_coordinate = final_coordinate[sorted_indicies]
    final_U = final_U[sorted_indicies]

    return final_coordinate, final_U, gain


final_coordinates, final_U, gain = find_varible_values(np.array([[0, 1], [1,2]]), np.array([0, 1, 2]), 2)
