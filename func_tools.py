#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ⚠️ 哪些测试过在稀疏矩阵下有效，要标注一下
import scipy as sp
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
    return sp.sparse.linalg.norm(matrix,axis)

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
    num_points = len(X)
    Xn = np.tile(X,(num_points,1,1))
    XnT = np.transpose(Xn,axes=(1,0,2))
    mat = Xn - XnT
    tool_mat = np.triu(-np.ones((num_points,num_points))) + np.tril(np.ones((num_points,num_points)))
    q = mat * tool_mat[:,:,np.newaxis]
    q_norm = np.linalg.norm(q,axis=2)
    mask = q_norm <= delta
    mask_1 = q_norm > delta
    q[mask] = q[mask]/delta
    q[mask_1] = q[mask_1]/q_norm[mask_1,np.newaxis]
    return (q*labda).sum(axis=0)+X-A


if __name__ == '__main__':
    A = np.array([12,24,10,0,0,0,0,0,0])
    sA = sp.sparse.csr_matrix(A)
    test_sparse(norm,sA) # True