# -*- coding: utf-8 -*-
"""AGN Simulation plotter

This script loads a qdp file and generates a scatter plot from the headers provided.

Example usage:
python plotter.py

Saves a plot in an outdir

"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
from statsmodels.stats.outliers_influence import summary_table

plt.style.use('publication.mplstyle')
OUTDIR = "outdir"


def read_qdp_file(fname: str) -> pd.DataFrame:
    """ Reads qdp file of 2 cols and returns x, y data
    :param fname str: path of file to be read
    :return: pd.DataFrame with x,y data
    """
    contents = open(fname).readlines()
    row_id_to_skip = [id for id, line in enumerate(contents) if line[0] != ' ']
    return pd.read_csv(fname, sep='\s+', skiprows=row_id_to_skip, names=['x', 'y'])


def plot_scatter(data: pd.DataFrame, xlabel: str, ylabel: str):
    """Plots a scatter plot using the dataframe's {x,y}."""
    fname = os.path.join(OUTDIR, "scatter.png")
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
    ax.plot(data.x, data.y, '.')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(fname)


def density_estimation(x, y, xmin, xmax, ymin, ymax):
    """Creates a density estimate for x,y data (useful for contour plotting)"""
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


def plot_contour(data: pd.DataFrame, xlabel: str, ylabel: str):
    """Plots a contour and plot using the dataframe's {x,y}."""
    fname = os.path.join(OUTDIR, "contour.png")
    xmin, xmax = 0, 1
    X, Y, Z = density_estimation(data.x, data.y, xmin, xmax, xmin, xmax)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cmap = "Blues"
    ax.imshow(np.rot90(Z), cmap=cmap, extent=[xmin, xmax, xmin, xmax])
    plt.contour(X, Y, Z, cmap=cmap)  # Add contour lines
    ax.plot(data.x, data.y, 'k.', markersize=2)
    plt.tight_layout()
    fig.savefig(fname)


def extrapolate_lines(x, y, new_x):
    f = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    return f(new_x)


def plot_custom_fit(data, xlabel, ylabel):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)

    x_fit = sm.add_constant(data.x)
    fit_results = sm.OLS(data.y, x_fit).fit()
    c = fit_results.params.const,
    m = fit_results.params.x
    st, fit_d, ss2 = summary_table(fit_results, alpha=0.05)
    predict_mean_ci_low, predict_mean_ci_upp = fit_d[:, 4:6].T
    new_x = np.linspace(-1, 1, num=300)
    fit_y = m * new_x + c
    ci_top = extrapolate_lines(data.x, predict_mean_ci_low, new_x)
    ci_bot = extrapolate_lines(data.x, predict_mean_ci_upp, new_x)

    ax.plot(data.x, data.y, '.', c='tab:blue', markersize=2, zorder=-1, alpha=0.5, label="Data")
    ax.plot(new_x, fit_y, color="tab:blue", zorder=0, alpha=0.7)
    ax.fill_between(new_x, ci_top, ci_bot, color="tab:blue", alpha=0.1, zorder=-10)

    fname = os.path.join(OUTDIR, "fitted_scatter_custom.png")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fit_label = "${y}={m:.2f}{x}+{c:.2f}$".format(
        y=ylabel.strip('$'),
        x=xlabel.strip('$'),
        c=fit_results.params.const,
        m=fit_results.params.x
    )
    ax.plot([], [], label=fit_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize='small', fancybox=False)
    plt.tight_layout()
    plt.savefig(fname)


def plot_fit(data: pd.DataFrame, xlabel: str, ylabel: str):
    """Plots a scatter and fit using the dataframe's {x,y}."""
    x_fit = sm.add_constant(data.x)
    fit_results = sm.OLS(data.y, x_fit).fit()
    print(fit_results.summary())
    fname = os.path.join(OUTDIR, "fitted_scatter.png")
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, xlabel=xlabel, ylabel=ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(data.x, data.y, '.', c='tab:blue', markersize=2, zorder=-1, alpha=0.5, label="Data")
    sns.regplot(
        x=data.x, y=data.y, x_estimator=np.mean, ax=ax, ci=95, x_bins=8, truncate=False, color="tab:blue"
    )
    fit_label = "${y}={m:.2f}{x}+{c:.2f}$".format(
        y=ylabel.strip('$'),
        x=xlabel.strip('$'),
        c=fit_results.params.const,
        m=fit_results.params.x
    )
    ax.plot([], [], label=fit_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize='small', fancybox=False)
    plt.tight_layout()
    plt.savefig(fname)


def main():
    data = read_qdp_file("data/anti_p15_t_op.qdp")
    kwargs = dict(data=data, xlabel=r"$\chi_{\rm{eff}}$", ylabel=r"$q$")
    # plot_scatter(**kwargs)
    # plot_contour(**kwargs)
    # plot_fit(**kwargs)
    # simple_regplot(data.x, data.y)
    plot_custom_fit(**kwargs)
    plt.show()


if __name__ == "__main__":
    main()
