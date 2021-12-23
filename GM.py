#!/usr/bin/env python
# -*- coding: utf-8 -*-

from func_tools import *

def backtrack(X,func,gX,d,alpha=1,sigma=0.1):
    right = func(X) + alpha*sigma*gX.T.dot(d)
    while func(X+alpha*d) > right:
        alpha = alpha * sigma
    return alpha

def GM(X,func,grad,tol):
    # 使用 tol 平方，这样只要算一次平方，不用每次都计算梯度的开方
    tol = tol ** 2
    # grad(X) 预算好，避免判断和更新要计算两次梯度
    gX = grad(X)
    while norm2(gX) > tol:
        step_size = backtrack(X,func,gX,-gX)
        X = X - step_size * gX
        gX = grad(X)
    return X