#!/usr/bin/env python
# -*- coding: utf-8 -*-

from func_tools import *

step_size = lambda n,lbd,delta:1/(1+n*lbd/delta)

# AGM 对稀疏和稠密是一样的
def AGM(grad,x0,step_size,tol):
    t_0 = t = 1
    x = x_0 = x0
    B = gen_B(x.shape[0])
    D = B.T 
    loss = norm(grad(x,B,D))
    losses = [loss]
    while loss > tol:
        # get beta_k (beta, t均是标量)
        beta = (t_0 - 1) / t
        t_0 = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_0**2))
        # update x
        y = x + beta * (x - x_0)
        x_0 = x
        ngrad= grad(y,B,D)
        x = y - step_size * ngrad
        loss = norm(ngrad)
        losses.append(loss)
    return x,losses

def AGM_weighted(grad,x0,step_size,tol,k,constant):
    t_0 = t = 1
    x = x_0 = x0
    B = gen_B(x.shape[0],sparse=False)
    W = gen_W(x,k,constant)
    loss = norm(grad(x,B,W))
    losses = [loss]
    while loss > tol:
        # get beta_k (beta, t均是标量)
        beta = (t_0 - 1) / t
        t_0 = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_0**2))
        # update x
        y = x + beta * (x - x_0)
        x_0 = x
        ngrad= grad(y,B,W)
        x = y - step_size * ngrad
        loss = norm(ngrad)
        losses.append(loss)
        # print(loss)
    return x,losses