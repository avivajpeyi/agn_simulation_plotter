# -*- coding: utf-8 -*-
"""AGN Simulation plotter

This script loads a qdp file and generates a scatter plot from the headers provided.

Example usage:
python plotter.py

Saves a plot in an outdir

"""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon

plt.style.use('publication.mplstyle')
OUTDIR = "outdir"

DATA = {
    "a_bulk": "data/r7x100_out_qx.qdp",
    "a_trap": "data/r7x100_qx_trap.qdp",
    "b_bulk": "data/anti_p15_b_op.qdp",
    "b_trap": "data/anti_p15_t_op.qdp",
    "b_contours": "data/linedata.json",
}

KWARGS = dict(
    bulk=dict(color="tab:red"),
    trap=dict(color="tab:blue"),
    contour=dict(color="tab:gray")
)

import numpy as np


class SimData:
    def __init__(self, label, bulk, trap, contour=""):
        self.label = label
        self.bulk = read_qdp_file(bulk)
        self.trap = read_qdp_file(trap)
        if contour:
            self.contour_dat = read_contour_file(contour)

    def add_scatter_to_plot(self, ax, extra_kwargs, scalex):
        kwargs = dict(zorder=-5, **extra_kwargs)
        ax.scatter(self.trap.xeff, self.trap.q, **kwargs, color='tab:red', marker="s")
        ax.scatter(self.bulk.xeff, self.bulk.q, **kwargs, color='tab:blue', marker="o")
        ax.annotate(self.label, xy=get_axis_limits(ax, scalex=scalex), fontsize="large", weight='bold')


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
    return pd.read_csv(fname, sep='\s+', skiprows=row_id_to_skip, names=['xeff', 'q'])


def get_axis_limits(ax, scalex=0.88, scaley=0.88):
    return ax.get_xlim()[1] * scalex, ax.get_ylim()[1] * scaley


def add_regions(ax):
    xeffmin, xeffmax = -1, 1
    qmin, qmax = 0, 1
    polys = []
    solid_kwarg = dict( zorder=-10, alpha=0.15)
    line_kwarg = dict(lw=2, fill=False, zorder=-1)

    # 0 < xeff <= 0.5 and q < 0.3, mid-bot
    polys.append(Polygon(
        xy=[(0, qmin), (0.5, qmin), (0.5, 0.3), (0, 0.3)],
        color="tab:green", **solid_kwarg)
    )
    polys.append(Polygon(
        xy=[(0, qmin), (0.5, qmin), (0.5, 0.3), (0, 0.3)],
        color="tab:green", **line_kwarg)
    )

    # xeff < 0, left
    polys.append(Polygon(
        xy=[(xeffmin, qmin), (0, qmin), (0, qmax), (xeffmin, qmax)],
        color="tab:purple", **solid_kwarg)
    )
    polys.append(Polygon(
        xy=[(xeffmin, qmin), (0, qmin), (0, qmax), (xeffmin, qmax)],
        color="tab:purple", **line_kwarg, ls='--')
    )

    # xeff > 0.3 and q > 0.5, top-right
    polys.append(Polygon(
        xy=[(0.3, 0.5), (xeffmax, 0.5), (xeffmax, qmax), (0.3, qmax)],
        color="tab:orange", **solid_kwarg)
    )
    polys.append(Polygon(
        xy=[(0.3, 0.5), (xeffmax, 0.5), (xeffmax, qmax), (0.3, qmax)],
        color="tab:orange", **line_kwarg, ls=':')
    )

    for poly in polys:
        ax.add_patch(poly)

    ax.annotate('Region 1', (-0.9, 0.9))
    ax.annotate('Region 2', (0.31, 0.52))
    ax.annotate('Region 3', (0.15, 0.06),arrowprops=dict(arrowstyle="->"),xytext=(-0.9, 0.05))


def add_contour(ax, df):
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


def plot_scatter(data_a, data_b, xlabel: str, ylabel: str):
    """Plots a scatter plot using the dataframe's {x,y}."""
    fname = os.path.join(OUTDIR, "scatter.png")
    fig, axes = plt.subplots(1, 2, figsize=(8, 8.0 / 2.0))

    for i, ax in enumerate(axes):
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, rotation=0, labelpad=10.0)
        if i == 0:
            ax.set_xlim(-1, 1)
        else:
            ax.set_xlim(-0.2, 1)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75])

    add_regions(axes[0])
    data_a.add_scatter_to_plot(axes[0], extra_kwargs=dict( s=0.1, alpha=0.5), scalex=0.75)

    add_contour(axes[1], data_b.contour_dat)
    data_b.add_scatter_to_plot(axes[1], extra_kwargs=dict(s=3, alpha=1), scalex=0.85)


    plt.tight_layout()
    fig.savefig(fname)


def main():
    data_a = SimData(label=r"$\bf{(a)}$", bulk=DATA['a_bulk'], trap=DATA['a_trap'])
    data_b = SimData(label=r"$\bf{(b)}$", bulk=DATA['b_bulk'], trap=DATA['b_trap'], contour=DATA['b_contours'])
    kwargs = dict(data_a=data_a, data_b=data_b, xlabel=r"$\chi_{\rm{eff}}$", ylabel=r"$q$")
    plot_scatter(**kwargs)


if __name__ == "__main__":
    main()
