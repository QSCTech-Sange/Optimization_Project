#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 所有的画图都放在这里
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")

def plot_generated_data(centroids, points, N, palette='muted'):

    begin = np.pad(N.cumsum(), (1,0))
    colors = sns.color_palette(palette,n_colors=len(centroids))

    for i in range(len(centroids)):
        plt.scatter(points[begin[i]:begin[i+1],0], points[begin[i]:begin[i+1],1], color=colors[i], label='cluster %d'%(i+1),alpha=0.8)
        plt.scatter(centroids[i][0], centroids[i][1], color=colors[i], marker='x',label='centroid %d'%(i+1))
    
    plt.show()

def plot_res_data(points,result,palette='muted'):
    colors = sns.color_palette(palette,n_colors=len(points))
    for i in range(len(points)):
        plt.scatter(points[i][0],points[i][1],color=colors[i])
        plt.scatter(result[i][0],result[i][1],color=colors[i],marker='x')

if __name__ == '__main__':
    pass