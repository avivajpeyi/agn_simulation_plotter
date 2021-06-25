# -*- coding: utf-8 -*-
"""Module one liner

This module does what....

Example usage:

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import Uniform, Normal
from corner import hist2d
from scipy import stats

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


def density_estimation(x, y, xmin, xmax, ymin, ymax):
    """Creates a density estimate for x,y data (useful for contour plotting)"""
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


def plot():
    x_len = 10.0
    fig = plt.figure(figsize=(x_len / 2.0, x_len / 2.0))
    ax = fig.add_subplot(111, xlim=(-1, 1), ylim=(0, 1))
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
    fig.savefig(f"{OUTDIR}/predictions.png")


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


def get_line_data(m, b, jitter=0.23, min_x=0.0, max_x=0.6, num=1000):
    jit = Uniform(minimum=-jitter, maximum=jitter).sample(num)
    xeff_samp = Uniform(minimum=min_x, maximum=max_x).sample(num)
    q_samp = (m * xeff_samp) + b + jit
    return pd.DataFrame(dict(x=xeff_samp, y=q_samp))


def plot_rhs_data(ax, c):
    data = []
    linex, liney = [0, 0.6], [1, 0.2]
    m = (liney[1] - liney[0]) / (linex[1] - linex[0])
    data.append(get_line_data(m=m, b=1, min_x=0.111, max_x=0.555, num=4000))
    data.append(get_normal_data(x=0.111, y=0.855, scale=0.025, num=1000))
    data.append(get_normal_data(x=0.555, y=0.259, scale=0.015, num=1000))
    data.append(get_line_data(m=m, b=1, min_x=0.66, max_x=1, num=50, jitter=0.0001))
    data = pd.concat(data, ignore_index=True)
    data = format_data(data, [0, 1], [0, 1])
    hist2d(data.x.values, data.y.values, ax=ax, new_fig=False, color=c, **HIST_KWARGS,
           contour_kwargs=dict(alpha=0.1, linewidths=0.1))


def plot_lhs_data(ax, c):
    data = []
    m = (0.6 - 1.0) / (-0.8 - 0.0)
    data.append(get_line_data(m=m, b=1, min_x=-0.777, max_x=-0.0222, num=4000))
    data.append(get_line_data(m=m, b=1, min_x=-0.98, max_x=-0.8, num=100, jitter=0.00001), )
    data.append(get_normal_data(x=-0.666, y=0.666, scale=0.02, num=1000))
    data = pd.concat(data, ignore_index=True)
    data = format_data(data, [-1, 0], [0, 1])
    hist2d(data.x.values, data.y.values, ax=ax, new_fig=False, color=c, **HIST_KWARGS,
           contour_kwargs=dict(alpha=0.1, linewidths=0.1))


def add_data_to_plot(ax):
    rhs_col, lhs_col = "tab:blue", "tab:red"
    plot_lhs_data(ax, lhs_col)
    plot_rhs_data(ax, rhs_col)


def set_xy_lables(ax, xlabel, ylabel):
    xlim, ylim = ax.get_xlim()[1], ax.get_ylim()[1]
    ax.annotate(xlabel, xy=(xlim * 1.05, ylim * 0), annotation_clip=False, fontsize='xx-large')
    ax.annotate(ylabel, xy=(xlim * 0.0, ylim * 1.05), annotation_clip=False, fontsize='xx-large')


def set_ticks(ax):
    xtk_labels = [-1, 0, 0.6, 1]
    ytk_labels = [0.2, 1]

    xlim, ylim = ax.get_xlim()[1], ax.get_ylim()[1]
    plt.xticks([])  # labels
    plt.yticks([])

    for xtick_label in xtk_labels:
        ax.annotate(xtick_label, xy=(xtick_label, ylim * -0.15), annotation_clip=False, ha='center', fontsize='x-large')
    for ytick_label in ytk_labels:
        ax.annotate(ytick_label, xy=(xlim * -0.15, ytick_label), annotation_clip=False, va='center', fontsize='x-large')
    ax.xaxis.set_ticks_position('none')  # tick markers
    ax.yaxis.set_ticks_position('none')
    add_axes_line(ax, xticks=[-1, 1], yticks=[1], xmin_ticks=[0.6], ymin_ticks=[0.2])


def remove_axes_splines(ax):
    ax.grid(False)
    # removing the default axis on all sides:
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)


def add_axes_line(ax, xticks, yticks, xmin_ticks, ymin_ticks):
    line_kwargs = dict(lw=1.3, c='k', zorder=10, clip_on=False)

    ax.axvline(0, ymin=0, ymax=1, **line_kwargs)
    ax.axhline(0, xmin=0, xmax=1, **line_kwargs)
    for x in xticks:
        ax.plot([x, x], [-0.05, 0.05], **line_kwargs)
    for y in yticks:
        ax.plot([-0.05, 0.05], [y, y], **line_kwargs)
    for x in xmin_ticks:
        ax.plot([x, x], [-0.025, 0.025], **line_kwargs)
    for y in ymin_ticks:
        ax.plot([-0.025, 0.025], [y, y], **line_kwargs)


def add_arrows_to_axes(fig, ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1. / 20. * (ymax - ymin)
    hl = 1. / 20. * (xmax - xmin)
    lw = 1.  # axis line width
    ohg = 0.3  # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw,
             head_width=hw, head_length=hl, overhang=ohg,
             length_includes_head=True, clip_on=False)

    ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw,
             head_width=yhw, head_length=yhl, overhang=ohg,
             length_includes_head=True, clip_on=False)


def main():
    plot()


if __name__ == "__main__":
    main()
