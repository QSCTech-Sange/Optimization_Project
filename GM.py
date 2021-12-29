#!/usr/bin/env python
# -*- coding: utf-8 -*-

from func_tools import *
import pandas as pd

def backtrack(X,func,gX,d,B,alpha=1,gamma=0.01,sigma=0.5):
    right_1 = func(X,B)
    right_2 = gamma*mat2vec(gX).dot(mat2vec(d))
    while func(X+alpha*d,B) > right_1 + right_2 * alpha:
        alpha = alpha * sigma
    return alpha

def BB(X,X_1,gX,gX_1):
    sk_1 = mat2vec(X) - mat2vec(X_1)
    y_1 = mat2vec(gX) - mat2vec(gX_1)
    alpha = sk_1.T.dot(sk_1) / (sk_1.T.dot(y_1))
    return alpha

def GM(X,func,grad,tol,watch=False):
    B = gen_B(X.shape[0])
    D = B.T
    gX = grad(X,B,D)
    norm_ = norm(gX)
    loss = [func(X,B)] ##修改了loss function的定义
    while norm_ > tol:
        step_size = backtrack(X,func,gX,-gX,B)
        X = X - step_size * gX
        gX = grad(X,B,D)
        norm_ = norm(gX)
        loss.append(func(X,B)) ##修改了loss function的定义
        if watch:
            print(norm_)
    return X,loss

def backtrack_Weighted(X,func,gX,d,B,nW,alpha=1,gamma=0.01,sigma=0.5):
    right_1 = func(X,B,nW)
    right_2 = gamma*mat2vec(gX).dot(mat2vec(d))
    while func(X+alpha*d,B,nW) > right_1 + right_2 * alpha:
        alpha = alpha * sigma
    return alpha

def GM_Weighted(X,func,grad,tol,k,ctt,watch=False):
    B = gen_B(X.shape[0])
    W = gen_W(X,k,constant=ctt,sparse=True)
    nW = gen_nw(X,k,ctt,sparse=True)
    gX = grad(X,B,W)
    norm_ = norm(gX)
    losses = [norm_]
    while norm_ > tol and len(losses) < 10000:
        step_size = backtrack_Weighted(X,func,gX,-gX,B,nW)
        X = X - step_size * gX
        gX = grad(X,B,W)
        norm_ = norm(gX)
        losses.append(norm_)
        if watch:
            print(lennorm_)
    return X,losses

def GM_BB(X,func,grad,tol):
    B = gen_B(X.shape[0])
    D = B.T
    gX = grad(X,B,D)
    iter = 0
    norm_2 = norm2(gX)
    loss = [func(X,B)] ##修改了loss function的定义
    tol = tol ** 2
    while norm_2 > tol:
        if iter ==0:
            step_size = backtrack(X,func,gX,-gX,B)
        else:
            step_size = BB(X,X_1,gX,gX_1)
        X_1 = X    
        X = X - step_size * gX
        gX_1 = grad(X_1,B,D)
        gX = grad(X,B,D)
        norm_2 = norm2(gX)
        loss.append(func(X,B)) ##修改了loss function的定义
        iter = iter + 1
    return X,loss