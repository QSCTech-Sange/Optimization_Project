#!/usr/bin/env python
# -*- coding: utf-8 -*-

from func_tools import *
import pandas as pd

def backtrack(X,func,gX,d,alpha=1,gamma=0.01,sigma=0.5):
    right = func(X) + alpha*gamma*mat2vec(gX.T).dot(mat2vec(d))
    while func(X+alpha*d) > right:
        alpha = alpha * sigma
    return alpha

def GM(X,func,grad,tol):
    gX = grad(X)
    norm_2 = norm2(gX)
    loss = [norm_2]
    tol = tol ** 2
    while norm_2 > tol:
        step_size = backtrack(X,func,gX,-gX)
        X = X - step_size * gX
        gX = grad(X)
        norm_2 = norm2(gX)
        loss.append(norm_2)
    return X,loss