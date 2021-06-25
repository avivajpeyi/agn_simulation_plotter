# -*- coding: utf-8 -*-
"""AGN Simulation plotter

This script loads a qdp file and generates a scatter plot from the headers provided.

Example usage:
python plotter.py

Saves a plot in an outdir

"""
import os
from string import ascii_lowercase

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('publication.mplstyle')
OUTDIR = "outdir"

import numpy as np

def read_qdp_file(fname: str) -> pd.DataFrame:
    """ Reads qdp file of 2 cols and returns x, y data
    :param fname str: path of file to be read
    :return: pd.DataFrame with x,y data
    """
    contents = open(fname).readlines()
    row_id_to_skip = [id for id, line in enumerate(contents) if line[0] != ' ']
    return pd.read_csv(fname, sep='\s+', skiprows=row_id_to_skip, names=['x', 'y'])


def get_axis_limits(ax, scalex=0.88, scaley=0.88):
    return ax.get_xlim()[1] * scalex, ax.get_ylim()[1] * scaley

@np.vectorize
def region_1(x, y):
    if x <= 0:
        return 1
    else:
        return 0

@np.vectorize
def region_2(x, y):
    if x > 0.3 and y > 0.5:
        return 1
    else:
        return 0

@np.vectorize
def region_3(x, y):
    if 0 < x <= 0.3 and y < 0.3:
        return 1
    else:
        return 0




def add_contour_for_region(ax, r_condtion, kwargs):
    xs = np.linspace(-1, 1, 100)
    ys = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xs, ys)
    Z = r_condtion(X, Y)
    cs = ax.contourf(X, Y, Z, [0.5, 1], **kwargs)
    cmap = cs.cmap.copy()
    cmap.set_under("#00000000")


def plot_scatter(data, xlabel: str, ylabel: str):
    """Plots a scatter plot using the dataframe's {x,y}."""
    fname = os.path.join(OUTDIR, "scatter.png")
    fig, axes = plt.subplots(1, 2, figsize=(8, 8.0 / 2.0))
    for i, ax in enumerate(axes):
        l = r"$\bf{(" + ascii_lowercase[i] + r")}$"
        plt_kwargs = dict(marker='.', zorder=-1, alpha=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, rotation=0, labelpad=10.0)
        ax.set_ylim(0, 1)
        if i == 0:
            d = data[i]
            ax.set_xlim(-1, 1)
            plt_kwargs.update(dict(s=0.5, alpha=0.75))
            ax.annotate(l, xy=get_axis_limits(ax, scalex=0.75), fontsize="large", weight='bold')
            ax.scatter(d.x, d.y, **plt_kwargs)
            add_contour_for_region(ax, region_1, kwargs=dict(colors=["tab:red"], alpha=0.3, zorder=-10))
            add_contour_for_region(ax, region_2, kwargs=dict(colors=["tab:orange"], alpha=0.3, zorder=-10))
            add_contour_for_region(ax, region_3, kwargs=dict(colors=["tab:green"], alpha=0.3, zorder=-10))
        else:
            ax.set_xlim(0, 1)
            ax.annotate(l, xy=get_axis_limits(ax), fontsize="large", weight='bold')
            ax.scatter(data[i].x, data[i].y, **plt_kwargs, c="tab:blue")

    plt.tight_layout()
    fig.savefig(fname)


def main():
    data_a = read_qdp_file("data/r7x100_out_qx.qdp")
    data_b = read_qdp_file("data/anti_p15_t_op.qdp")
    kwargs = dict(data=[data_a, data_b], xlabel=r"$\chi_{\rm{eff}}$", ylabel=r"$q$")
    plot_scatter(**kwargs)


if __name__ == "__main__":
    main()
