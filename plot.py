#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 所有的画图都放在这里
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")

def plot_generated_data(centroids, points, N, palette='muted'):

    begin = np.pad(N.cumsum(), (1,0), mode='constant')
    colors = sns.color_palette(palette,n_colors=len(centroids))

    for i in range(len(centroids)):
        plt.scatter(points[begin[i]:begin[i+1],0], points[begin[i]:begin[i+1],1], color=colors[i], label='cluster %d'%(i+1),alpha=0.8)
        plt.scatter(centroids[i][0], centroids[i][1], color=colors[i], marker='x',label='centroid %d'%(i+1))
    
    plt.show()

# 绘制分组的结果
# 参数 way = ans 的时候，只绘制结果，不绘制簇心
# 参数 way = points 的时候，只绘制簇心，不绘制结果
def plot_res_data(points,result,groups,palette='muted',way='all'):
    if type(points) != np.ndarray:
        points = np.array(points.todense())
    colors = sns.color_palette(palette,n_colors=np.unique(groups).shape[0])
    for i in range(points.shape[0]):
        if way!='ans':
            plt.scatter(points[i][0],points[i][1],color=colors[groups[i]])
        if way!='points':
            plt.scatter(result[i][0],result[i][1],color=colors[groups[i]],marker='x')

if __name__ == '__main__':
    pass