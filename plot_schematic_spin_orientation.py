# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import matplotlib.pyplot as plt
import pandas as pd
from bilby.core.prior import Uniform, Normal
from corner import hist2d

plt.style.use('publication.mplstyle')
OUTDIR = "outdir"

HIST_KWARGS = dict(
    smooth=3,
    truth_color="tab:orange",
    plot_density=False,
    plot_datapoints=False,
    fill_contours=True,
    plot_contours=True,
    alpha=0.3,
    zorder=-10,
)


def plot():
    x_len = 10.0
    fig = plt.figure(figsize=(x_len/2.0 , x_len / 2.0))
    ax = fig.add_subplot(111, xlim=(0, 5), ylim=(0, 5))
    add_data_to_plot(ax)
    xlab, ylab = r"$\chi_{\rm{eff}}$", r"$q$"
    simple = True
    if simple:
        set_xy_lables(ax, xlab, ylab)
        remove_axes_splines(ax)
        set_ticks(ax)
        # add_arrows_to_axes(fig, ax)
    else:
        ax.set_xlabel(xlab, fontsize='xx-large')
        ax.set_ylabel(ylab, rotation=0, labelpad=10.0, fontsize='xx-large')
        ax.axvline(0, c='k', lw=1.5)

    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/sigmas.png")


def get_normal_data(x, y, scale, num=100):
    xeff_samp = Normal(mu=x, sigma=scale).sample(num)
    q_samp = Normal(mu=y, sigma=scale).sample(num)
    return pd.DataFrame(dict(x=xeff_samp, y=q_samp))


def format_data(data, xrange, yrange):
    data = data[data['y'] > yrange[0]]
    data = data[data['y'] < yrange[1]]
    data = data[data['x'] > xrange[0]]
    data = data[data['x'] < xrange[1]]
    return data

def main():
    plot()


if __name__ == "__main__":
    main()
