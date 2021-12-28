from func_tools import *
import pandas as pd


def backtrack(X,func,gX,d,B,alpha=1,gamma=0.01,sigma=0.5):
    right =  gamma*mat2vec(gX).T.dot(mat2vec(d))  ## first vec than T
    ## right should not include alpha
    while func(X+alpha*d,B) - func(X,B)> alpha * right: 
        alpha = alpha * sigma
    return alpha

def BFGS(X,func,grad,tol,p=1):
    B = gen_B(len(X))
    D = B.T
    n = len(X)
    d = len(X[0])
    H = p*np.eye(n*d)
    gX = grad(X,B,D)
    norm_2 = norm(gX)
    tol = tol**2
    loss = [func(X,B)]
    while(norm_2 > tol):
        dk = - H.dot(mat2vec(gX))
        dk = vec2mat(dk,n,d)
        step_size = backtrack(X,func,gX,dk,B)
        X_1 = X
        X = X + step_size*dk
        gX_1 = grad(X_1,B,D)
        gX = grad(X,B,D)
        norm_2 = norm(gX)
        loss.append(func(X,B))
        print("nomr_2:", norm_2)
        s = mat2vec(X - X_1)
        y = mat2vec(gX - gX_1)
        if(s.T.dot(y))<0:
            H = H 
        else:
            w = s - H.dot(y)
            H = H + ((np.outer(w,s) + np.outer(s,w)) / s.T.dot(y)) - (w.T.dot(y) / (s.T.dot(y))**2 ) * np.outer(s,s)
        print("s.T.dot(y):",s.T.dot(y))
    return X, loss
        
        