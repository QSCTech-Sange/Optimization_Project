#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 所有的画图都放在这里

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")

def plot_generated_data(centroids, points, N, palette='dark'):

    begin = N.cumsum()
    colors = sns.color_palette(palette,n_colors=len(centroids))

    for i in range(len(centroids)):
        if i==0:
            plt.scatter(points[:begin[i], 0], points[:begin[i], 1], color=colors[i], alpha=0.3)
        else:
            plt.scatter(points[begin[i-1]:begin[i], 0], points[begin[i-1]:begin[i], 1], color=colors[i], alpha=0.3)
        plt.scatter(centroids[i][0], centroids[i][1], color=colors[i], marker='x')
    
    plt.show()

if __name__ == '__main__':
    pass