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
def norm(matrix):
    if type(matrix)==np.ndarray:
        return np.linalg.norm(matrix)
    return sp.sparse.linalg.norm(matrix)

# 高效范数方
# 已经验证过是相等的了
def norm2(matrix):
    if type(matrix)==np.ndarray:
        return (matrix**2).sum()
    return matrix.power(2).sum()

# huber = lambda x,delta: np.where(norm(x)<=1,0.5*norm2(x)/delta,norm(x)-0.5*delta)
# grad_huber = lambda x,delta: np.where(norm(x)<=delta,x/delta,x/norm(x))
# step_size = lambda x,n,delta,lamda : 1/(1+n*lamda/delta)


if __name__ == '__main__':
    A = np.array([12,24,10,0,0,0,0,0,0])
    sA = sp.sparse.csr_matrix(A)
    test_sparse(norm,sA) # True