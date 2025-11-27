#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:47:15 2020

@author: ashish
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy 
import scipy.linalg

t = np.linspace(0.01, 1, 100)
#t = np.random.random_sample(size = 100)
b = np.sin(10*t)

m = 15

A = np.zeros((len(t), m))

count = 0
for i in t:
    for j in range(m):
        A[count, j] = i**j
    
    count += 1

plt.figure()
plt.scatter(t, b, c = 'red')
plt.title('Original Figure')

def modified_QR(matrix):
    rows, cols = matrix.shape
    Q = np.zeros((rows, cols))
    R = np.zeros((cols, cols))
    V = np.zeros((rows, cols))
    #q = []
    
    for i in range(cols):
        V[:, i] = matrix[:, i]
    
    for i in range(cols):
        R[i, i] = np.linalg.norm(V[:, i])
        Q[:, i] = V[:, i] / R[i, i]
        
        for j in range(i+1, cols):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j]*Q[:, i]
        
    return Q, R

def backsubstitution(matrix, vector):
    matrix = np.array(matrix)
    rows, cols = matrix.shape
    X = np.zeros(cols)
    
    for i in range(cols - 1, -1, -1):
        temp_sum = 0
        for j in range(i+1, cols):
            temp_sum += matrix[i, j]*X[j]
        X[i] = (vector[i] - temp_sum) / matrix[i, i]
        
    return X
    
b = b.reshape((-1, 1))
gram_Q, gram_R = modified_QR(A)
gram_Q_t_b = np.matmul(gram_Q.T, b)
gram_X = backsubstitution(gram_R, gram_Q_t_b)
gram_sin = np.matmul(A, gram_X)

plt.figure()
plt.scatter(t, gram_sin, c = 'green')
plt.title('Modified Gram Schmidt method')

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1
    
def householder_R(matrix):
    matrix = copy.deepcopy(matrix)
    rows, cols = matrix.shape
    v = [None]*cols
    for k in range(cols):
        x = matrix[k : rows, k].reshape(-1, 1)
        #print(x.shape)
        unit_vector = np.array([1] + [0]*(x.shape[0] - 1)).reshape(-1, 1)
        v[k] = sign(x[0, 0])*np.linalg.norm(x)*unit_vector + x
        v[k] = v[k] / np.linalg.norm(v[k])
        
        matrix[k : rows, k : cols] = matrix[k : rows, k : cols] - np.matmul(2*v[k], np.matmul(v[k].T, matrix[k : rows, k : cols]))

    return matrix, v

house_R, house_v = householder_R(A)


def householder_Qb(house_v, b):
    cols = len(house_v)
    rows = b.shape[0]
    #b = np.ravel(b)
    for k in range(cols):
        #house_v[k] = np.ravel(house_v[k])
        #print(b[k : rows].shape)
        b[k : rows] =  b[k : rows] - 2*np.dot(house_v[k].T, b[k : rows])*house_v[k]
    return b    
    
house_Q_t_b = householder_Qb(house_v, b)

house_X = backsubstitution(house_R, house_Q_t_b)
house_sin = np.matmul(A, house_X)

plt.figure()
plt.scatter(t, house_sin, c = 'blue')
plt.title('Householder Triangulization Method')



def SVD(matrix, b):
    
    U, sigma, V_t = np.linalg.svd(matrix, full_matrices = False)
    sigma = np.diagflat(sigma)
    #print(sigma)
    #print(U.shape, sigma.shape, V_t.shape)
    U_t_b = np.matmul(U.T, b)
    #print(U_t_b.shape)
    SVD_w = backsubstitution(sigma, U_t_b)
    #print(SVD_w.shape)
    svd_x = np.matmul(V_t.T, SVD_w)
    #print(svd_x.shape)
    
    return svd_x


svd_X = SVD(A, b)
SVD_sin = np.matmul(A, svd_X)
SVD_sin = np.ravel(SVD_sin)

plt.figure()
plt.scatter(t, SVD_sin, c = 'black')
plt.title('SVD based method')


#print(np.diagflat([1,2,2,4]))

# In-built Function
np_lstsqr = np.linalg.lstsq(A, b)[0]
np_sin = np.matmul(A, np_lstsqr)

plt.figure()
plt.scatter(t, np_sin, c = 'grey')
plt.title('Numpy Library Based Method')


def normal_equation(matrix, b):
    
    A_t_b = np.matmul(matrix.T, b)
    A_t_A = np.matmul(matrix.T, matrix)
    #print(np.linalg.matrix_rank(A_t_A))
    #L = np.linalg.cholesky(A_t_A)
    #w = scipy.linalg.solve_triangular(L, A_t_b)
    x = np.matmul(np.linalg.inv(A_t_A), A_t_b)
    
    return x

normal_X = normal_equation(A, b)
normal_sin = np.matmul(A, normal_X)

plt.figure()
plt.scatter(t, normal_sin, c = 'orange')
plt.title('Normal Equation based method')

#print(np.linalg.matrix_rank(np.matmul(A.T, A)))





# -------------------- Question 4 --------------------------------------

def LU_without_pivot(A):
    U = A
    rows, cols = A.shape
    L = np.identity(rows)
    for i in range(rows - 1):
        for j in range(i + 1, rows):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i : rows] = U[j, i : rows] - L[j, i]*U[i, i : rows]
    return L, U        
        
n = []
norm_pivot = []
norm_without_pivot = []
for N in range(5, 21):
    A = np.random.rand(N, N)
    A = A - np.diagflat(np.diag(A)) + np.diagflat(0.001 * np.ones(N))
    
    # LU factorization with pivoting
    
    sci_p, sci_l, sci_u = scipy.linalg.lu(A)
    lu_a = np.matmul(sci_l, sci_u) - A
    norm = np.linalg.norm(lu_a, ord = 'fro')
    
    without_l, without_u = LU_without_pivot(A)
    lu_a_without = np.matmul(without_l, without_u) - A
    norm_without = np.linalg.norm(lu_a_without, ord = 'fro')
    
    n.append(N)
    norm_pivot.append(norm)
    norm_without_pivot.append(norm_without)

plt.figure()
plt.scatter(n, norm_pivot, c = 'blue', label = 'Partial Pivoting')
plt.scatter(n, norm_without_pivot, c = 'red', label = 'Without pivot')
plt.xlabel('N')
plt.ylabel('Frobenius Norm of LU - A')
plt.legend()












