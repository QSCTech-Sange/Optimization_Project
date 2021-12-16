# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:37:20 2021

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt

def Generate_Points(centroids, p, nplist, sigma_2): #generate points based on p centroids
    a = []
    for i in range(len(centroids)):
        for j in range(nplist[i]):
            epsilon = np.random.normal(loc=0, scale=sigma_2[i], size=2)
            a.append(centroids[i]+epsilon)
    return np.array(a)

def Centroids(p): #generate p centroids
    c = np.random.randn(p,2)*10
    return c

def Data_Preparation(low, high, p): 
    #low: the minimum number of points in a cluster; high: the maximum number of points in a cluster (unattainable)
    nplist = np.random.randint(low, high, p)
    sigma_2 = np.random.rand(p)**2 * 5 #variance
    centroids = Centroids(p)
    a = Generate_Points(centroids, p, nplist, sigma_2)
    return a, nplist, centroids

sample, np, c = Data_Preparation(50, 100, 7)
print(len(sample))
print(np)
loc = 0
#plot the sample points
for num in np:
    points = sample[loc: loc + num]
    plt.scatter(points[:,0],points[:,1])
    loc += num
#plot the centroids
for i in range(len(np)):
    plt.scatter(c[i][0],c[i][1], c='r', marker = 'x')
plt.show()