#!/usr/bin/env python
# -*- coding: utf-8 -*-

from func_tools import *


def step_size(n, lbd, delta): return 1/(1+n*lbd/delta)

# backtracking


def bck(func, grad, X, step_size, gamma, sigma):
    d = -grad(X)
    while func(X + step_size*d) > func(X) - gamma*step_size*(np.linalg.norm(d)**2):
        step_size *= sigma
    return step_size


# 还未验证对稀疏和稠密是否一样
def GM(func, grad, stepsize_method, x0, step_size0, tol):
    time0 = time.time()
    df = pd.DataFrame(columns={"f_value", "step_size", "gradient"})
    step_size = step_size0
    x = x0
    loss = norm(grad(x))

    while loss > tol:
        step_size = stepsize_method(x, step_size)
        x -= step_size*grad(x)
        loss = norm(grad(x))
        f_value = func(x)
        df = df.append(
            [{"f_value": f_value, "step_size": step_size, "gradient": loss}])
    print("Used seconds:", time.time()-time0)
    return x

# AGM 对稀疏和稠密是一样的


def AGM(func, grad, x0, step_size, tol):
    time0 = time.time()
    df = pd.DataFrame(columns={"f_value", "beta", "t", "loss"})
    t_0 = t = 1
    x = x_0 = x0
    loss = norm(grad(x))

    while loss > tol:
        f_value = func(x)
        loss = norm(grad(x))

        # get beta_k (beta, t均是标量)
        beta = (t_0 - 1) / t
        t_0 = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_0**2))
        # update x
        y = x + beta * (x - x_0)
        x_0 = x
        x = y - step_size * grad(y)
        df = df.append(
            [{"f_value": f_value, "beta": beta, "t": t, "loss": loss}])
    print(df)
    print("Used seconds:", time.time()-time0)
    return x
