import os 
import os.path as osp
import math
import re
import tqdm
import datetime 
import sys
import json

import pandas as pd
import numpy as np 
import pickle as pkl
import pandas as pd

import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.cm as cm 
import seaborn as sns

class GifMaker():
    def __init__(self):
        self.datas = []

    def add_data(self, d):
        self.datas += [d]

    def add_datas(self, ds):
        self.datas += ds

    def make(self, titles=None, folder="./", file_name="Non", save=True, show=False):
        datas_np = np.array(self.datas)
        max_value = np.nanmax(datas_np)
        min_value = np.nanmin(datas_np)
        print(f'max: {max_value}')
        print(f'min: {min_value}')

        def make_heatmap(i):
            ax.cla()
            if titles:
                ax.set_title(titles[i])
            else:
                ax.set_title("Iteration="+str(i))
            data = np.array(self.datas[i])
            sns.heatmap(data, ax=ax, cbar=True, cbar_ax=cbar_ax, vmin=min_value, vmax=max_value)
            ax.set_aspect('equal', adjustable='box')
        #fms = len(self.datas) if len(self.datas)<=128 else np.linspace(0, len(self.datas)-1, 128).astype(int)
        fms = len(self.datas) 
        grid_kws = {'width_ratios': (0.9, 0.05), 'wspace': 0.2}
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw = grid_kws, figsize = (12, 8))
        ani = animation.FuncAnimation(fig=fig, func=make_heatmap, frames=fms, interval=500, blit=False)
        if save:
            file_path = osp.join(folder, file_name+".gif")
            ani.save(file_path, writer="pillow")
        if show:
            plt.show() 
        plt.close()

    def reset(self):
        plt.close()
        self.datas = []

def plot_datas(datas, labels, use_axis=False, n_rowcol=(1, 4)):
    n_data = len(datas)
    if n_rowcol[0]*n_rowcol[1] < n_data:
        n_col = n_rowcol[1]
        n_row = n_data//n_col+1
    else:
        n_row = n_rowcol[0]
        n_col = n_rowcol[1]

    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*5, n_row*5))
    if n_row==1:
        for n in range(n_col):
            axes[n].axis('off')
            im = axes[n].imshow(datas[n])
            cbar = fig.colorbar(im, ax=axes[n])
            axes[n].set_title(labels[n])

    for r in range(n_row):
        for c in range(n_col):
            if n_col*r+c >= n_data:
                axes[r, c].axis('off')
                continue
            if not use_axis:
                axes[r, c].axis('off')
            im = axes[r, c].imshow(datas[n_col*r+c])
            cbar = fig.colorbar(im, ax=axes[r, c])
            axes[r, c].set_title(labels[n_col*r+c])
    plt.show()



def multi_render(datas, filename, labels, n_rowcol=(None, None), vmin=None, vmax=None, cmap='viridis', use_kde=False):
    # datas shape = (n_datas, n_times, data row, data col)
    n_datas = len(datas)
    if not n_rowcol[0]:
        n_rowcol = (1, n_datas)
    n_row = n_rowcol[0]
    n_col = n_rowcol[1]
        

    fig, axes = plt.subplots(n_row, n_col+1, figsize = (5*n_row, 5*n_col))
    if not vmin or not vmax:
        vmax = np.nanmax(datas)
        vmin = np.nanmin(datas)

    if n_row==1:
        im = axes[n_col-1].imshow(datas[n_datas-1][0], vmin=vmin, vmax=vmax, cmap=cmap) 
        fig.colorbar(im, ax=axes[n_col])
    else:
        im = axes[n_row-1, n_col-1].imshow(datas[n_datas-1][0], vmin=vmin, vmax=vmax, cmap=cmap) 
        fig.colorbar(im, ax=axes[0, n_col])

    for i in range(n_row*(n_col+1)):
        idx = i if n_row==1 else (i//(n_col+1),  i-(n_col+1)*(i//(n_col+1)))
        axes[idx].axis('off')
        
    imgs = []
    contours = []
    for n in range(n_datas):
        idx = n if n_row==1 else (n//n_col,  n-n_col*(n//n_col))
        axes[idx].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        im = axes[idx].imshow(datas[n][0], vmin=vmin, vmax=vmax, cmap=cmap, animated=True) 
        imgs.append(im)
        if use_kde:
            X, Y, Z, _ = calc_kde(datas[n][0])
            Y = -Y + 9 
            cs = axes[idx].contour(Y, X, Z, 10, animated=True)
            contours.append(cs)

    def animate(i, imgs, contours, datas):
        for n in range(n_datas):
            idx = n if n_row==1 else (n//n_col,  n-n_col*(n//n_col))
            axes[idx].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            #axes[n].imshow(datas[n][i], vmin=vmin, vmax=vmax, cmap=cmap) 
            imgs[n].set_array(datas[n][i])

            if use_kde:
                for c in contours[n].collections:
                    c.remove()
                X, Y, Z, _ = calc_kde(datas[n][i])
                Y = -Y + 9 
                contours[n] = axes[idx].contour(Y, X, Z, 10)
        for i in range(n_row):
            idx = n_col if n_row==1 else (i,  n_col)
            axes[idx].axis('off')

        for i in range(n_datas, n_row*n_col):
            idx = i if n_row==1 else (i//n_col,  i-n_col*(i//n_col))
            axes[idx].axis('off')

        return imgs, contours 

    ani = animation.FuncAnimation(fig, animate, fargs=(imgs, contours, datas), frames=range(len(datas[0])), blit=False, interval = 200)

    for i in range(n_datas):
        idx = i if n_row==1 else (i//n_col,  i-n_col*(i//n_col))
        axes[idx].set_title(labels[i])

    path = filename
    print(path)
    ani.save(path, writer="ffmpeg", fps=5)
    plt.close()
    print(f"Save {path}")

        
    fig, axes = plt.subplots(n_row, n_col+1, figsize = (5*n_row, 5*n_col))
    vmax = np.nanmax(datas)
    vmin = np.nanmin(datas)

    if n_row==1:
        im = axes[n_col-1].imshow(datas[n_datas-1][0], vmin=vmin, vmax=vmax, cmap=cmap) 
        fig.colorbar(im, ax=axes[n_col])
    else:
        im = axes[n_row-1, n_col-1].imshow(datas[n_datas-1][0], vmin=vmin, vmax=vmax, cmap=cmap) 
        fig.colorbar(im, ax=axes[0, n_col])

    for i in range(n_row*(n_col+1)):
        idx = i if n_row==1 else (i//(n_col+1),  i-(n_col+1)*(i//(n_col+1)))
        axes[idx].axis('off')
        
    imgs = []
    contours = []
    for n in range(n_datas):
        vmax = np.nanmax(datas[n][0])
        vmin = np.nanmin(datas[n][0])
        if vmax>np.abs(vmin):
            vmin = -np.abs(vmax)
        else:
            vmax = np.abs(vmin)
        idx = n if n_row==1 else (n//n_col,  n-n_col*(n//n_col))
        axes[idx].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
        im = axes[idx].imshow(datas[n][0], vmin=vmin, vmax=vmax, cmap=cmap, animated=True) 
        imgs.append(im)
        if use_kde:
            X, Y, Z, _ = calc_kde(datas[n][0])
            Y = -Y + 9 
            cs = axes[idx].contour(Y, X, Z, 10, animated=True)
            contours.append(cs)

    def animate(i, imgs, contours, datas):
        for n in range(n_datas):
            vmax = np.nanmax(datas[n][i])
            vmin = np.nanmin(datas[n][i])
            if vmax>np.abs(vmin):
                vmin = -np.abs(vmax)
            else:
                vmax = np.abs(vmin)
            idx = n if n_row==1 else (n//n_col,  n-n_col*(n//n_col))
            axes[idx].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
            #axes[n].imshow(datas[n][i], vmin=vmin, vmax=vmax, cmap=cmap) 
            imgs[n].set_array(datas[n][i])

            if use_kde:
                for c in contours[n].collections:
                    c.remove()
                X, Y, Z, _ = calc_kde(datas[n][i])
                Y = -Y + 9 
                contours[n] = axes[idx].contour(Y, X, Z, 10)
        for i in range(n_row):
            idx = n_col if n_row==1 else (i,  n_col)
            axes[idx].axis('off')

        for i in range(n_datas, n_row*n_col):
            idx = i if n_row==1 else (i//n_col,  i-n_col*(i//n_col))
            axes[idx].axis('off')

        return imgs, contours 

    ani = animation.FuncAnimation(fig, animate, fargs=(imgs, contours, datas), frames=range(len(datas[0])), blit=False, interval = 200)

    for i in range(n_datas):
        idx = i if n_row==1 else (i//n_col,  i-n_col*(i//n_col))
        axes[idx].set_title(labels[i])

    fig, axes = plt.subplots(n_row, n_col+1, figsize = (5*n_row, 5*n_col))
    path =filename[:-4] + 'vimin-max' + filename[-4:]
    ani.save(path, writer="ffmpeg", fps=5)
    plt.close()
    print(f"Save {path}")