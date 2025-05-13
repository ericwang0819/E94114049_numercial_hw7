# -*- coding: utf-8 -*-
"""
Created on Tue May 13 22:55:43 2025

@author: ericd
"""

import numpy as np

# 定義係數矩陣 A 和常數向量 b
A = np.array([
    [4, -1,  0, -1,  0,  0],
    [-1, 4, -1,  0, -1,  0],
    [0, -1, 4,  0,  1, -1],
    [-1, 0,  0,  4, -1, -1],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# Jacobi Method
def jacobi_method(A, b, x0=None, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)

    for _ in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# Gauss-Seidel Method
def gauss_seidel_method(A, b, x0=None, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# SOR Method (Successive Over-Relaxation)
def sor_method(A, b, omega=1.1, x0=None, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = x[i] + omega * ((b[i] - s1 - s2) / A[i, i] - x[i])
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# Conjugate Gradient Method
def conjugate_gradient_method(A, b, x0=None, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = np.dot(r, r)

    for _ in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            return x
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

# 解方程組
x_jacobi = jacobi_method(A, b)
x_gs = gauss_seidel_method(A, b)
x_sor = sor_method(A, b)
x_cg = conjugate_gradient_method(A, b)

# 顯示結果
print("Jacobi Method:        ", np.round(x_jacobi, 6))
print("Gauss-Seidel Method:  ", np.round(x_gs, 6))
print("SOR Method:           ", np.round(x_sor, 6))
print("Conjugate Gradient:   ", np.round(x_cg, 6))
