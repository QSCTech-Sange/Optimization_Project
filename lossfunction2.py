# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from func_tools import *

def isAmong(i, j, A, k):
    s = []
    for input in A:
            s.append((norm(i - input), input))
    s = sorted(s)
    knn = [x[1] for x in s[1:(k + 1)]]
    return (j in knn)

def weight(i, j, constant, k, A):
    if isAmong(i, j, A, k):
        return np.exp(-constant * norm(i - j) ** 2)
    else:
        return 0

def loss_func2(X,A,lbd,B,k,constant):
    q = B.dot(X)
    pairs = []
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            pairs.append((i,j))
    w = np.array(list(weight(A[x[0]], A[x[1]], constant, k, A) for x in pairs))
    q = w * q * lbd
    return 0.5*norm2(X-A) + q 

def grad_func2(X,delta,A,lbd,B,k,constant):
    pairs = []
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            pairs.append((i,j))
    w = np.array(list(weight(A[x[0]], A[x[1]], constant, k, A) for x in pairs))
    return X - A + lbd * B.T * w