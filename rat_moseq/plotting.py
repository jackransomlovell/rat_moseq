import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import scipy
import h5py
import cv2

from toolz import merge
import platform


def format_plots():
    '''
    Defines a series of formatting options for plots and applies them globally.
    '''
    all_fig_dct = {
        "pdf.fonttype": 42,
        "figure.figsize": (3, 3),
        # "font.family": "sans-serif",
        # "font.sans-serif": "Helvetica",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        'savefig.facecolor': 'white',
        'savefig.transparent': True,
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        "axes.labelcolor": "black",
        "text.color": "black",
        'xtick.color': 'black',
        'ytick.color': 'black',
        'svg.fonttype': 'none',
        'lines.linewidth': 1,
        'axes.linewidth': 0.5,
    }

    # all in points
    font_dct = {
        "axes.labelpad": 2.5,
        "font.size": 6,
        "axes.titlesize": 6,
        "axes.labelsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "xtick.major.size": 1.75,
        "ytick.major.size": 1.75,
        "xtick.minor.size": 1.75,
        "ytick.minor.size": 1.75,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.pad": 1,
        "ytick.major.pad": 1,
        "xtick.minor.pad": 1,
        "ytick.minor.pad": 1,
    }

    plot_config = merge(all_fig_dct, font_dct)

    plt.style.use('default')
    for k, v in plot_config.items():
        plt.rcParams[k] = v
    sns.set_style('white', merge(sns.axes_style('ticks'), plot_config))
    sns.set_context('paper', rc=plot_config)

    if platform.system() != 'Darwin':
        plt.rcParams['ps.usedistiller'] = 'xpdf'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['figure.dpi'] = 200

def imshow(ax, data, cmap, label, vmin=0, vmax=310):
    im = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        interpolation='none'
    )
    ax.set(
        yticks=[],
        xticks = np.linspace(0, data.shape[1], 50).astype(int)
          )
    return im

def plot_state_frame(flattened, proj, spine_height, cp, fontsize=12, vmin=0, vmax=310):

    fig, ax = plt.subplots(4, 2, figsize=(8, 3.5), gridspec_kw={'hspace': .05, 'wspace': 0.00001, 'width_ratios': [1, 0.05]})
    
    pix = imshow(ax[0, 0], flattened, 'cubehelix', 'Changepoints', vmin=vmin, vmax=vmax)
    spine = imshow(ax[1, 0], spine_height, 'cubehelix', 'spine \n height', vmin=vmin, vmax=vmax)
    rand = imshow(ax[2, 0], proj, 'RdBu_r', 'Projection', vmin=-1.5, vmax=1.5)

    ax[0, 0].margins(x=0)
    
    
    cbar = fig.colorbar(pix, ax=ax[1, 1], location='right', fraction=.95, shrink=1.85, aspect=35, ticks = [vmin, vmax], anchor=(1.25, -.125))
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.yaxis.set_tick_params(direction='out', length=1.25)  # Optionally adjust padding
    cbar.set_label('mm', labelpad=0.9, **{'size': fontsize})
    
    rand_cbar = fig.colorbar(rand, ax=ax[2, 1], location='right', fraction=.95, shrink=.9, ticks=[-1.5, 1.5], anchor=(1.25, .75))
    rand_cbar.ax.yaxis.set_ticks_position('left')
    rand_cbar.ax.tick_params(labelsize=fontsize)
    rand_cbar.ax.yaxis.set_tick_params(direction='out', length=1.25)  # Optionally adjust padding
    rand_cbar.set_label('loading', labelpad=0.9, **{'size': fontsize})
    
    
    ax[1, 0].set_ylabel('spine \n height', labelpad=.5)
    ax[1,0 ].margins(x=0)
    ax[1,0 ].set_xticks([])
    
    ax[0, 0].set_xticks([])
    ax[0, 0].set_ylabel('raw \n pixels', labelpad=.5)
    ax[0, 0].margins(x=0)
    
    ax[2, 0].set_xticks([0, 30])
    ax[2, 0].set_xticks([])
    ax[2, 0].set_ylabel('rand. \n project.', labelpad=.5)
    ax[2, 0].margins(x=0)
    
    ax[3, 0].plot(cp, color='k')
    ax[3, 0].set_ylabel('normalized \n change \n score')
    ax[3, 0].margins(x=0)
    ax[3, 0].set_xticks([])
    ax[3, 0].set_yticks([])
    
    for _ax in ax[:, 0]:
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.spines['bottom'].set_visible(False)
        _ax.spines['left'].set_visible(False)
    
    ax[3, 0].spines['bottom'].set_visible(True)
    ax[3, 0].spines['left'].set_visible(True)
    
    for _ax in ax[:, 1]:
        _ax.axis('off')
    
    return fig, ax