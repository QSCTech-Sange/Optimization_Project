

from func_tools import *
import pandas as pd



def backtrack(X,func,gX,d,B,alpha=1,gamma=0.01,sigma=0.5):
    right =  gamma*mat2vec(gX).T.dot(mat2vec(d))  ## first vec than T
    ## right should not include alpha
    func_value = func(X,B)
    while func(X+alpha*d,B) - func_value> alpha * right: 
        alpha = alpha * sigma
        print("alpha:" , alpha)
    return alpha

def LBFGS(X,func,grad,tol,p=1,m=5):
    B = gen_B(len(X))
    D = B.T
    n = len(X)
    d = len(X[0])
    H = p*np.eye(n*d)
    gX = grad(X,B,D)
    dk = - H.dot(mat2vec(gX))
    norm_2 = norm(gX)
    tol = tol**2
    loss = [func(X,B)]

    iter = 0
    y_ls = []
    s_ls = []
    while(norm_2 > tol):

        dk = vec2mat(dk,n,d)
        
        step_size = backtrack(X,func,gX,dk,B)
        print("step_size:", step_size)
        X_1 = X
        X = X + step_size*dk
        gX_1 = grad(X_1,B,D)
        gX = grad(X,B,D)
        norm_2 = norm(gX)
        loss.append(func(X,B))
        s = mat2vec(X - X_1)
        y = mat2vec(gX - gX_1)

        if(iter<m):
            y_ls.append(y)
            s_ls.append(s)
        else:
            y_ls.pop(0)
            y_ls.append(y)
            s_ls.pop(0)
            s_ls.append(s)
        
        if(s.dot(y))<0:
            H = H 
        else:
            q = mat2vec(gX)
            H = (s.dot(y) / y.dot(y)) * np.eye(n*d)
            for i in range(len(y_ls)-1,0,-1):
                alpha = (s_ls[i].T.dot(q))/(s_ls[i].T.dot(y_ls[i]))
                q = q - alpha * y_ls[i]
            r = H.dot(q)
            for i in range(len(y_ls)):
                beta = y_ls[i].T.dot(r) / (s_ls[i].T.dot(y_ls[i]))
                alpha = (s_ls[i].T.dot(q)) / (s_ls[i].T.dot(y_ls[i])) 
                r = r + (alpha - beta) * s_ls[i]
        dk = -r
        
            

        print("s.T.dot(y):",s.dot(y))
        print("Iter:",iter)
        print("norm_2: ",norm_2)

        iter = iter + 1
    return X, loss
        
        