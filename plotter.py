# -*- coding: utf-8 -*-
"""AGN Simulation plotter

This script loads a qdp file and generates a scatter plot from the headers provided.

Example usage:
python plotter.py

Saves a plot in an outdir

"""
import json
import os
from string import ascii_lowercase

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

plt.style.use('publication.mplstyle')
OUTDIR = "outdir"

DATA = {
    "a_bulk": "data/r7x100_out_qx.qdp",
    "a_trap": "data/r7x100_qx_trap.qdp",
    "b_bulk": "data/anti_p15_b_op.qdp",
    "b_trap": "data/anti_p15_t_op.qdp",
    "b_contours": "data/linedata.json",
}

import numpy as np

class SimData:
    def __init__(self, label, bulk, trap, contour=""):
        self.label = label
        self.bulk = read_qdp_file(bulk)
        self.trap = read_qdp_file(trap)
        if contour:
            self.contour_dat = read_contour_file()



def read_contour_file(fname):
    """Conotur data stored in json with data stored in columns with
    'qs, chiEff_mean_plus_one_sigma, chiEff_mean_minus_one_sigma'
    """
    with open(fname) as f:
        data = json.load(f)
    qs = data['qs']
    mean_xeff_plus_1sig = np.mean(data['chiEff_mean_plus_one_sigma'], axis=0)
    mean_xeff_minus_1sig = np.mean(data['chiEff_mean_minus_one_sigma'], axis=0)
    return pd.DataFrame(dict(
        q=qs,
        xeff_plus=mean_xeff_plus_1sig,
        xeff_minus=mean_xeff_minus_1sig,
    ))


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
    if 0 < x <= 0.5 and y < 0.3:
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


def plot_lhs_scatter(ax, d, plt_kwargs, l):
    ax.set_xlim(-1, 1)
    d_1g = d[(d['x'] <= 0.25) & (d['x'] >= -0.25)]
    d_2g = d[~((d['x'] <= 0.25) & (d['x'] >= -0.25))]
    ax.scatter(d_1g.x, d_1g.y, **plt_kwargs, color='tab:blue', marker=',', s=0.1, alpha=0.3)
    ax.scatter(d_2g.x, d_2g.y, **plt_kwargs, color='tab:gray', marker=',', s=0.1, alpha=0.8)
    add_contour_for_region(ax, region_1, kwargs=dict(colors=["tab:purple"], alpha=0.4, zorder=-10))
    add_contour_for_region(ax, region_2, kwargs=dict(colors=["tab:orange"], alpha=0.25, zorder=-10))
    add_contour_for_region(ax, region_3, kwargs=dict(colors=["tab:green"], alpha=0.25, zorder=-10))
    ax.annotate(l, xy=get_axis_limits(ax, scalex=0.75), fontsize="large", weight='bold')


def add_fit(data: pd.DataFrame, ax):
    """Plots a scatter and fit using the dataframe's {x,y}."""
    x_fit = sm.add_constant(data.x)
    fit_results = sm.OLS(data.y, x_fit).fit()
    # ax.plot(data.x, data.y, '.', c='tab:blue', markersize=2, zorder=-1, alpha=0.5, label="Data")
    sns.regplot(
        x=data.x, y=data.y, ax=ax, ci=95, truncate=False, color="tab:blue", scatter=False
    )
    fit_label = "$q={m:.2f}xeff+{c:.2f}$".format(
        c=fit_results.params.const,
        m=fit_results.params.x
    )
    print(fit_label)


def plot_rhs_plot(ax, d, plt_kwargs, l):
    ax.set_xlim(-0.2, 1)
    ax.plot(d.x, d.y, '.', c='tab:blue', markersize=2, zorder=-1, alpha=0.5, label="Data")
    add_fit(d, ax)
    plot_rhs_contour(ax)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75])

    ax.annotate(l, xy=get_axis_limits(ax, scalex=0.85), fontsize="large", weight='bold')


def plot_rhs_contour(ax):
    df = read_contour_file()
    zord = -30
    kwargs = dict(color="tab:gray", linestyle="dashed", lw=1.5, zorder=zord, alpha=0.5)
    ax.plot(df.xeff_plus, df.q, **kwargs)
    ax.plot(df.xeff_minus, df.q, **kwargs)
    ax.fill_betweenx(
        df.q,
        df.xeff_plus,
        df.xeff_minus,
        interpolate=True,
        facecolor="tab:gray",
        alpha=0.2,
        zorder=zord
    )


def plot_scatter(data, xlabel: str, ylabel: str):
    """Plots a scatter plot using the dataframe's {x,y}."""
    fname = os.path.join(OUTDIR, "scatter.png")
    fig, axes = plt.subplots(1, 2, figsize=(8, 8.0 / 2.0))
    for i, ax in enumerate(axes):
        l = r"$\bf{(" + ascii_lowercase[i] + r")}$"
        plt_kwargs = dict(zorder=-1)
        ax.set_ylim(0, 1)
        d = data[i]
        if i == 0:
            plot_lhs_scatter(ax, d, plt_kwargs, l)
        else:
            plot_rhs_plot(ax, d, plt_kwargs, l)
            plot_rhs_contour(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, rotation=0, labelpad=10.0)

    plt.tight_layout()
    fig.savefig(fname)


def main():
    data_a = read_qdp_file("data/r7x100_out_qx.qdp")
    data_b = read_qdp_file("data/anti_p15_t_op.qdp")
    kwargs = dict(data=[data_a, data_b], xlabel=r"$\chi_{\rm{eff}}$", ylabel=r"$q$")
    plot_scatter(**kwargs)


if __name__ == "__main__":
    main()
