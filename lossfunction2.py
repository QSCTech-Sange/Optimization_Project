# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from func_tools import *

def isAmong(i, j, X, k):
    s = []
    for input in range(len(X)):
        s.append((norm(i - input), X[input]))
    s = sorted(s)
    knn = [x[1] for x in s[1:(k + 1)]]
    return (j in knn)

def weight(i, j, constant, k, X):
    if isAmong(i, j, X, k):
        return np.exp(-constant * norm(X[i] - X[j]) ** 2)
    else:
        return 0

def loss_func2(X,A,lbd,B,k,constant):
    q = B.dot(X)
    pairs = []
    for i in range(len(X) - 1):
        for j in range(i + 1, len(X)):
            pairs.append((i,j))
    w = np.array(list(weight(x[0], x[1], constant, k, X) for x in pairs))
    q = w * q * lbd
    return 0.5*norm2(X-A) + q 

def grad_func2(X,A,lbd,B,k,constant):
    pairs = []
    for i in range(len(X) - 1):
        for j in range(i + 1, len(X)):
            pairs.append((i,j))
    w = np.array(list(weight(X[x[0]], X[x[1]], constant, k, X) for x in pairs))
    return X - A + lbd * B.T * w

