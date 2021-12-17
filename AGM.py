#!/usr/bin/env python
# -*- coding: utf-8 -*-

from func_tools import *

step_size = lambda n,lbd,delta:1/(1+n*lbd/delta)

def AGM(grad,x0,step_size,tol):
    t_0 = t = 1
    x = x_0 = x0
    loss = norm(grad(x))

    while loss > tol:
        # get beta_k (beta是标量)
        beta = (t_0 - 1) / t
        t_0 = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_0**2))
        
        # update x
        y = x + beta * (x - x_0)
        x_0 = x
        x = y - step_size * grad(y)
        loss = norm(grad(x))

    return x