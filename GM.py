#!/usr/bin/env python
# -*- coding: utf-8 -*-

from func_tools import *
import pandas as pd

def backtrack(X,func,gX,d,B,alpha=1,gamma=0.01,sigma=0.5):
    right = func(X,B) + alpha*gamma*mat2vec(gX.T).dot(mat2vec(d))
    while func(X+alpha*d,B) > right:
        alpha = alpha * sigma
    return alpha

def BB(X,X_1,gX,gX_1):
    sk_1 = mat2vec(X) - mat2vec(X_1)
    y_1 = mat2vec(gX) - mat2vec(gX_1)
    alpha = sk_1.T.dot(sk_1) / (sk_1.T.dot(y_1))
    return alpha


def GM(X,func,grad,tol):
    B = gen_B(len(X))
    D = gen_D(len(X))
    gX = grad(X,B,D)
    norm_ = norm(gX)
    loss = [norm_]
    while norm_ > tol:
        step_size = backtrack(X,func,gX,-gX,B)
        X = X - step_size * gX
        gX = grad(X,B,D)
        norm_ = norm(gX)
        loss.append(norm_)
    return X,loss

def GM_BB(X,func,grad,tol):
    gX = grad(X)
    iter = 0
    norm_2 = norm2(gX)
    loss = [norm_2]
    tol = tol ** 2
    while norm_2 > tol:
        if iter ==0:
            step_size = backtrack(X,func,gX,-gX)
        else:
            step_size = BB(X,X_1,gX,gX_1)
        X_1 = X    
        X = X - step_size * gX
        gX_1 = grad(X_1)
        gX = grad(X)
        norm_2 = norm2(gX)
        loss.append(norm_2)
        iter = iter +1
    return X,loss,iter