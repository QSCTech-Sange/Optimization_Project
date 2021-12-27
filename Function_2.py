# -*- coding: utf-8 -*-

from func_tools import *

def isAmong(i, j, X, k):
    s = []
    for input in range(len(X)):
        s.append((norm(X[i] - X[input]), input))
    s = sorted(s)
    knn = [x[1] for x in s[1:(k + 1)]]
    return (j in knn)

def weight(i, j, constant, k, X):
    if isAmong(i, j, X, k):
        return np.exp(-constant * norm(X[i] - X[j]) ** 2)
    else:
        return 0

def loss_func2(X,A,lbd,B,k,constant):
    q = np.squeeze(norm(B.dot(X), axis = 1))
    pairs = []
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            pairs.append((i,j))
    w = np.array(list(weight(x[0], x[1], constant, k, A) for x in pairs))
    q = q * w * lbd
    q=sum(q)
    return 0.5*norm2(X-A) + q 

def grad2(X,A,lbd,B,k,constant):
    pairs = []
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            pairs.append((i,j))
    d = np.array(list(list(weight(x[0], x[1], constant, k, A) * (X[x[0]] - X[x[1]]) / norm(X[x[0]] - X[x[1]])) for x in pairs))
    q = B.T.dot(d) * lbd
    return X - A + q 
