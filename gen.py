#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

# Two ways of generating centroids
# p number of centroids, scale is the distance between the centroids
def gen_centroid_norm(rng,p,scale=10):
    return rng.randn(p,2)*scale

def gen_centroid_uniform(rng,p,scale=10):
    return rng.rand(p,2)*scale

# N是每个簇的点数，是一个array
def gen_N(rng,p,low=0,high=50):
    return rng.randint(low, high, size=p)

# generate points around the centroids
# 注意传递的sigma是标准差，不是方差, N是每个簇的点数，是一个array
def gen_points(rng, centroids, N, sigma=1):
    extend_centroids = np.repeat(centroids, N, axis=0)
    return extend_centroids + rng.randn(len(extend_centroids),2)*sigma

def get_O(N):
    X = np.linspace(-10,10,N*2)
    Y = (100-X**2)**0.5
    sub = np.linspace(-10,-7,N*2//5,dtype=np.int8)
    sub_ = np.linspace(7,10,N*2//5,dtype=np.int8)
    sub__ = np.linspace(-7,7,N*2//5,dtype=np.int8)
    a = np.c_[X[sub],Y[sub]]
    b = np.c_[X[sub_],Y[sub_]]
    c = np.c_[X[sub__],Y[sub__]]
    d = np.c_[X[sub],-Y[sub]]
    e = np.c_[X[sub_],-Y[sub_]]
    f = np.c_[X[sub__],-Y[sub__]]    
    return np.r_[a,b,c,d,e,f]

def get_P(N):
    X = np.full(N,20)
    Y = np.linspace(-10,10,N)
    base = np.c_[X,Y]
    x_1 = np.linspace(20,40,N)
    y_1 = np.full(N,10)
    top = np.c_[x_1,y_1]
    y_2 = np.full(N,1)
    bottom = np.c_[x_1,y_2]
    x_3 = np.full(N//2+1,40)
    y_3 = np.linspace(1,10,N//2+1)
    right = np.c_[x_3,y_3]
    return np.r_[base,top,bottom,right]

def get_T(N):
    x_1 = np.linspace(50,70,N)
    y_1 = np.full(N,10)
    top = np.c_[x_1,y_1]
    x_2 = np.full(N,60)
    y_2 = np.linspace(-10,10,N)
    bottom = np.c_[x_2,y_2]
    return np.r_[top,bottom]

def get_opt(N1=20,N2=8,N3=8):
    O = get_O(N1)
    P = get_P(N2)
    T = get_T(N3)
    centroids = np.r_[O,P,T]
    return centroids

'''
生成一系列数据，参数如下所示
@param
    rng_num:随机数种子
    p:簇的个数
    low:簇内点的个数的下限
    high:簇内点的个数的上限
    sigma:簇内点的标准差
    scale:簇的范围扩大几倍（随机数生成在0-1之间，默认乘了20）
    distribution:簇的生成方式，uniform为均匀分布，norm为正态分布
@return
    centroids:簇的坐标
    points:点的坐标
'''
def gen_data(rng_num=114514,p=10,low=3,high=20,sigma=1,scale=20,distribution='uniform'):
    rng = np.random.RandomState(rng_num)
    if distribution=='uniform':
        centroids = gen_centroid_uniform(rng,p,scale)
    elif distribution=='OPT':
        p = p * 4
        N1 = p//2
        N2 = p//4
        N3 = p - N1 - N2
        centroids = get_opt(N1,N2,N3)
    else:
        centroids = gen_centroid_norm(rng,p,scale)
    N = gen_N(rng,centroids.shape[0],low,high)
    points = gen_points(rng, centroids, N, sigma)
    return centroids,points, N

if __name__ == '__main__':
    centroids, points, N = gen_data()