#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import sparse

def load_wine():
    wine_data = scipy.io.loadmat('data/wine/wine_data.mat')
    wine_label = scipy.io.loadmat('data/wine/wine_label.mat')
    return sparse.COO.from_scipy_sparse(wine_data['A'].T),wine_label['b']

def load_wine_dense():
    data, label = load_wine()
    return np.asarray(data.todense()).T, label

def load_vowel():
    vowel_data = scipy.io.loadmat('data/vowel/vowel_data.mat')
    vowel_label = scipy.io.loadmat('data/vowel/vowel_label.mat')
    return vowel_data['A'],vowel_label['b']

def load_vowel_dense():
    data, label = load_vowel()
    return np.asarray(data.todense()).T, label

def load_segment():
    segment_data = scipy.io.loadmat('data/segment/segment_data.mat')
    segment_label = scipy.io.loadmat('data/segment/segment_label.mat')
    return segment_data['A'],segment_label['b']

def load_segment_dense():
    data, label = load_segment()
    return np.asarray(data.todense()).T, label

def load_mnist():
    mnist_data = scipy.io.loadmat('data/mnist/mnist_data.mat')
    mnist_label = scipy.io.loadmat('data/mnist/mnist_label.mat')
    return mnist_data['A'],mnist_label['b']

def load_mnist_dense():
    data, label = load_mnist()
    return np.asarray(data.todense()).T, label

if __name__ == '__main__':
    # wine, mnist, vowel, segment 可替换
    # 加载稀疏矩阵
    wine_data, wine_label = load_wine()
    # 加载稠密矩阵
    wine_data, wine_label = load_wine_dense()