#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ⚠️ 哪些测试过在稀疏矩阵下有效，要标注一下
import scipy as sp
import scipy.sparse as sps
import numpy as np

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
# 已经验证过是相等的了
# 当 axis = 2 的时候，给定三维矩阵（最内层表示x1一个向量），这样输出的结果是对每一个xi进行求norm
# eg. 当一个矩阵2x2，这四个项每一个都是向量，此时 norm(矩阵，axis=2)，返回的结果仍然是2*2的，且对应点的值是向量的norm
def norm(matrix,axis=None):
    if type(matrix)==np.ndarray:
        return np.linalg.norm(matrix,axis)
    return (matrix**2).sum(axis=axis)**0.5

# 高效范数方
# 已经验证过是相等的了
def norm2(matrix):
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
# ********这个目前只对稠密有效*********
# 它非常的复杂，我不认为有人能光看下面代码，不看注释，能知道这是啥
def grad_hub_matrix(X,delta,A,labda):
    n,d = X.shape
    if type(X)==np.ndarray:
        mat = X.reshape(1,n,d) - X.reshape(n,1,d)
    else:
        mat = X.reshape((1,n,d)) - X.reshape((n,1,d))
    tool_mat = np.triu(-np.ones((n,n))) + np.tril(np.ones((n,n)))
    q = mat * tool_mat[:,:,np.newaxis]
    q_norm = (q**2).sum(axis=2)**0.5
    q = np.where((q_norm <= delta)[:,:,np.newaxis],q/delta,q/q_norm[:,:,np.newaxis])
    q = q * tool_mat[:,:,np.newaxis]
    return (q*labda).sum(axis=0)+X-A
        
def mat2vec(mat):
    return mat.reshape(mat.shape[0]*mat.shape[1])

def vec2mat(vec,n,d):
    return vec.reshape((n,d))

# 给定计算出来的 X，返回每个点所属的group编号
# X 格式类似 [x1,x2,x3,...]，每个xi是一个向量
# 一个简单的 DFS
def get_group(ans,tol=0.01):
    if type(ans)!=np.ndarray:
        ans = ans.todense()
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

def loss_func(X,A,lbd,delta):
    n = len(X)
    Xn = np.tile(X,(n,1,1))
    XnT = np.transpose(Xn,axes=(1,0,2))
    mask = np.triu(np.ones((n,n)),1)
    q = (XnT - Xn) * mask[:,:,np.newaxis]
    q_norm = np.linalg.norm(q,axis=2)
    less = q_norm<=delta
    more = q_norm > delta
    q[less] = q_norm[less,np.newaxis]**2 / 2 / delta
    q[more] = q_norm[more,np.newaxis] - delta/2
    return 0.5 * norm2(X-A) + lbd * q.sum()

# 我愿称之为最强B矩阵生成法
def gen_B(n=5,sparse=False):
    Y = np.repeat(np.arange(n-1),np.arange(n-1,0,-1))
    X = np.arange(n*(n-1)//2)
    X = np.r_[X,np.arange(n*(n-1)//2)]
    Y = np.r_[Y,np.arange(1,(n-1)*n//2+1) - np.repeat(np.array(np.r_[0,np.arange(n-2,0,-1)]).cumsum(),np.arange(n-1,0,-1))]
    data = np.r_[np.ones(n*(n-1)//2,dtype=np.int8),-np.ones(n*(n-1)//2,dtype=np.int8)]
    B = sps.csr_matrix((data,(X,Y)),shape=(n*(n-1)//2,n))
    return B if sparse else B.toarray()

if __name__ == '__main__':
    A = np.array([12,24,10,0,0,0,0,0,0])
    sA = sp.sparse.csr_matrix(A)
    test_sparse(norm,sA) # True