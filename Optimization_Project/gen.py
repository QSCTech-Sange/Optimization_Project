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
    else:
        centroids = gen_centroid_norm(rng,p,scale)
    N = gen_N(rng,p,low,high)
    points = gen_points(rng, centroids, N, sigma)
    return centroids,points, N

if __name__ == '__main__':
    centroids, points, N = gen_data()