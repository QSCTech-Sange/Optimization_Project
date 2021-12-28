#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ⚠️ 哪些测试过在稀疏矩阵下有效，要标注一下
import scipy as sp
import scipy.sparse as sps
import numpy as np
from sklearn import metrics

# 验证一个函数是否对稀疏矩阵也有效
def test_sparse(func,sparse_data):
    dense_data = sparse_data.todense()
    dense_data = np.asarray(dense_data)
    sparse_res = func(sparse_data)
    dense_res = func(dense_data)
    if type(sparse_res) == type(dense_res):
        return np.allclose(func(sparse_data),func(dense_data))
    return np.allclose(np.asarray(func(sparse_data).todense()),func(dense_data))

# 对矩阵（向量）求Frobenius norm
# 不加 axis，是整个矩阵求norm（得到一个标量）
# 加了 axis=1，是每一行求norm，得到一个向量，但是转为了n行1列的矩阵
# !!!!!!!!!!!!!!!!!!
# norm会转换成非sparse，但是通常norm从数值上就不是sparse的，所以无所谓
def norm(matrix,axis=None):
    if axis:
        if type(matrix)==np.ndarray or type(matrix)==np.matrix:
            return np.linalg.norm(matrix,axis=1).reshape((-1,1))
        return sp.sparse.linalg.norm(matrix,axis=1).reshape((-1,1))
    else:
        if type(matrix)==np.ndarray or type(matrix)==np.matrix:
            return np.linalg.norm(matrix)
        return sp.sparse.linalg.norm(matrix)     

# 高效范数方
# 已经验证过是相等的了
# 返回的是标量！
def norm2(matrix,axis=None):
    if axis:
        if type(matrix) == np.matrix:
            return (np.array(matrix)**2).sum(axis=1).reshape((-1,1))
        if type(matrix)==np.ndarray:
            return (matrix**2).sum(axis=1).reshape((-1,1))
        return matrix.power(2).sum(axis=1).reshape((-1,1))       
    else:
        if type(matrix) == np.matrix:
            return (np.array(matrix)**2).sum()
        if type(matrix)==np.ndarray:
            return (matrix**2).sum()
        return matrix.power(2).sum()

# huber_norm 对于向量的梯度
# 已经验证过是相等的了
def grad_hub_vec(vec,delta):
    return vec/delta if norm(vec) <= delta else vec/norm(vec)

# 对 X（所有xi） 求导，X是一个包含所有xi的矩阵，xi=[...]为内层,即 n * d 的矩阵
# A 是初始点
# delta 是参数
def grad_hub_matrix(X,delta,A,labda,B,D):
    y = B.dot(X)
    ynorm = norm(y,axis=1) # not sparse anymore, which is a FEATURE
    ynorm[ynorm<=delta] = delta
    y = y/ynorm
    # if X is sparse, here return will be matrix. Convert to ndarray
    # note here our return is always dense.
    if type(X) == np.ndarray:
        return X - A + labda * D.dot(y)
    else:
        return np.array(X - A + labda * D.dot(y))
        
def mat2vec(mat):
    return mat.reshape(mat.shape[0]*mat.shape[1])

def vec2mat(vec,n,d):
    return vec.reshape((n,d))

# 给定计算出来的 X，返回每个点所属的group编号
# X 格式类似 [x1,x2,x3,...]，每个xi是一个向量
# 一个简单的 DFS
def get_group(ans,tol=0.01):
    if type(ans)!=np.ndarray and type(ans)!=np.matrix:
        ans = np.array(ans.todense())
    groups = np.arange(len(ans))
    visited = [False] * len(ans)
    group_count = -1
    tol = tol**2
    for i in range(len(ans)):
        if not visited[i]:
            group_count += 1
            groups[i] = group_count
            visited[i] = True
            arr = [i]
            while arr:
                node = arr.pop()
                for j in range(len(ans)):
                    if not visited[j] and norm2(ans[node]-ans[j]) <= tol:
                        arr.append(j)
                        groups[j] = groups[node]
                        visited[j] = True
    return groups

def loss_func(X,A,lbd,delta,B):
    q = B.dot(X)
    qnorm = norm(q,axis=1)
    q = np.where(qnorm > delta, qnorm-delta/2, qnorm**2/2/delta)
    q = lbd * q.sum()
    return 0.5*norm2(X-A) + q 

# 我愿称之为最强B矩阵生成法
def gen_B(n,sparse=True):
    Y = np.repeat(np.arange(n-1),np.arange(n-1,0,-1))
    X = np.arange(n*(n-1)//2)
    X = np.r_[X,np.arange(n*(n-1)//2)]
    Y = np.r_[Y,np.arange(1,(n-1)*n//2+1) - np.repeat(np.array(np.r_[0,np.arange(n-2,0,-1)]).cumsum(),np.arange(n-1,0,-1))]
    data = np.r_[np.ones(n*(n-1)//2,dtype=np.int8),-np.ones(n*(n-1)//2,dtype=np.int8)]
    B = sps.csr_matrix((data,(X,Y)),shape=(n*(n-1)//2,n))
    return B if sparse else B.toarray()

# 可能用不上了
# def gen_C(n,sparse=True):
#     Y = np.repeat(np.arange(n),n-1)
#     X = np.arange(n*(n-1))
#     q = np.tile(np.arange(n),reps=(n,1)).flatten()
#     p = np.arange(0,(n+1)*n,n+1)
#     Y = np.r_[Y,np.delete(q,p)]
#     X = np.r_[X,np.arange(n*(n-1))]
#     data = np.r_[np.ones(n*(n-1),dtype=np.int8),-np.ones(n*(n-1),dtype=np.int8)]
#     C = sps.csr_matrix((data,(X,Y)),shape=(n*(n-1),n))
#     return C if sparse else C.toarray()

# 我愿称之为最强D矩阵生成法
# def gen_D(n,sparse=True):
#     X = np.r_[np.repeat(np.arange(n),np.arange(n-1,-1,-1)),np.repeat(np.arange(1,n),np.arange(1,n))]
#     q = np.tile(np.arange(n-1,0,-1),reps=(n-1,1))
#     q[:,0] = np.arange(n-1)
#     q = np.tril(q)
#     q = np.tril(q.cumsum(axis=1))
#     Y = np.r_[np.arange((n)*(n-1)//2),0,q[q!=0]]
#     data = np.r_[np.ones((n)*(n-1)//2,dtype=np.int8),-np.ones(n*(n-1)//2,dtype=np.int8)]
#     C = sps.csr_matrix((data,(X,Y)),shape=(n,n*(n-1)//2))
#     return C if sparse else C.toarray()

# def gen_W(a,k,B,constant,sparse=False):
#     n = a.shape[0]
#     begin = (n-1) * (n-2) // 2
#     E = norm2(B.dot(a),axis=1).flatten()
#     D = B.T
#     D[D==-1] = 1
#     W = np.zeros((n,n*(n-1)//2))
#     for i in range(n):
#         Q = D[i]*E
#         Q2 = Q.argsort()[begin:begin+k]
#         mask = np.ones(Q.shape,dtype=np.bool)
#         mask[Q2] = False
#         Q[mask] = 0
#         Q[Q2] = np.exp(-constant * Q[Q2])
#         W[i] = Q
#     return sps.csr_matrix(W) if sparse else W

def gen_W(a,k,constant,sparse=False):
    n = a.shape[0]
    W = np.zeros((n,n*(n-1)//2))
    neighbor_dist = []
    for i in range(n):
        neighbor_dist.append([])
        for j in range(n):
            neighbor_dist[-1].append(norm2(a[i] - a[j]))
    neighbor_threshold = []
    for i in neighbor_dist:
        neighbor_threshold.append(sorted(i)[k])
    pick = []
    for i in range(n):
        for j in range(i+1,n):
            if neighbor_dist[i][j] <= neighbor_threshold[i]:
                pick.append((i,j))
    for i,j in pick:
        nj = n*i - i*(i+1)//2 + j - i - 1
        W[i][nj] = np.exp(-constant * norm2(a[i] - a[j]))
        W[j][nj] = -np.exp(-constant * norm2(a[i] - a[j]))
    return sps.csr_matrix(W) if sparse else W

def weight_grad(X,A,B,W,lbd):
    Q = B.dot(X)
    Q = Q / norm(Q,axis=1)
    return W.dot(Q)*lbd + X - A

def gen_nw(a,k,constant,sparse=False):
    n = a.shape[0]
    W = np.zeros(n*(n-1)//2)
    neighbor_dist = []
    for i in range(n):
        neighbor_dist.append([])
        for j in range(n):
            neighbor_dist[-1].append(norm2(a[i] - a[j]))
    neighbor_threshold = []
    for i in neighbor_dist:
        neighbor_threshold.append(sorted(i)[k])
    pick = []
    for i in range(n):
        for j in range(i+1,n):
            if neighbor_dist[i][j] <= neighbor_threshold[i]:
                pick.append((i,j))
    for i,j in pick:
        nj = n*i - i*(i+1)//2 + j - i - 1
        W[nj] = np.exp(-constant * norm2(a[i] - a[j]))
    return sps.csr_matrix(W) if sparse else W

def loss_func_weighted(X,A,lbd,B,nw):
    return (0.5 * norm2(X-A,axis=1)).sum() + lbd * nw.dot(norm(B.dot(X),axis=1)) 

def auto_group(ans,group_count):
    l=0
    r=999
    while l<r:
        mid = (l + r) / 2
        if len(np.unique(get_group(ans,mid))) < group_count: r = mid
        elif len(np.unique(get_group(ans,mid))) > group_count: l = mid
        else: return get_group(ans,mid)
    print("The Tolrence is", l)
    return get_group(ans,l)

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

if __name__ == '__main__':
    A = np.array([12,24,10,0,0,0,0,0,0])
    sA = sp.sparse.csr_matrix(A)
    test_sparse(norm,sA) # True