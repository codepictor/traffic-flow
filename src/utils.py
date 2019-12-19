import os
import os.path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_matrix(matrix, plot_name, annot, linewidths,
                xticklabels=None, yticklabels=None, xlabel=None, ylabel=None):
    """Plot matrix to a file as a heat map.

    Parameters
    ----------
    matrix : array-like
        Data to visualize as a heat map.
    plot_name : str
        Plot's title as well as name of the output file.
    annot : bool
        If True, write the data value in each cell.
    linewidths : bool
        Width of the lines that will divide each cell.
    xticklabels : iterable object, optional
        Names of ticks on the x-axis.
    yticklabels : iterable object, optional
        Names of ticks on the y-axis.
    xlabel : str, optional
        The label text on the x-axis.
    ylabel : str, optional
        The label text on the y-axis.
    """
    if len(matrix.shape) != 2:
        raise ValueError('Only 2-D arrays can be displayed as a heat map.')
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Data should have a square shape.')
    if not isinstance(plot_name, str):
        raise TypeError('Name of a heat map should be a string.')

    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(
        pd.DataFrame(index=yticklabels, data=matrix, columns=xticklabels),
        square=True, annot=annot, fmt=".1f",
        linewidths=linewidths, linecolor='black', rasterized=True, ax=ax
    )
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(plot_name, fontsize=24)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', plot_name + '.pdf'
        ),
        dpi=150, format='pdf'
    )
    plt.close(fig)

