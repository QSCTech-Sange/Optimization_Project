from func_tools import *


def NewtonCG(g, H, bck, x0, cg_max, cg_tol_k, tol):
    xk = x0
    gk = g(xk)
    gradnorm = norm(gk)
    while gradnorm > tol:
        dk = conjugate_gradient(gk, H, xk, cg_max, cg_tol_k)
        # 这里dk要重新变回87*2
        dk = dk.reshape(xk.shape)
        alpha = bck(xk, gk)
        xk = xk + alpha*dk
        gk = g(xk)
        gradnorm = norm(gk)
        # cg_tol_k = max(1,gradnorm**0.1)*gradnorm
    return xk

# CG method


def conjugate_gradient(gk, H, xk, cg_max, cg_tol_k):
    vj = 0
    #c2 = 0.1
    rj = mat2vec(gk)
    pj = -rj
    for j in np.arange(cg_max):
        quadratic = np.dot(pj.T, np.dot(H(xk), pj))
        if quadratic <= 0:
            return -gk if j == 0 else vj
        # 和alpha类似
        sigmaj = norm2(rj)/quadratic
        vj1 = vj + sigmaj * pj
        rj1 = rj + sigmaj*np.dot(H(xk), pj)
        beta_j1 = norm2(rj1) / norm2(rj)  # 近似计算
        pj1 = -rj1 + beta_j1 * pj
        if norm(rj1) < cg_tol_k:
            dk = vj1
            return dk
        # 更新pj,vj,rj
        pj = pj1
        vj = vj1
        rj = rj1
        print(norm(vj1), norm(pj1), norm(rj1), quadratic)
    return vj1
