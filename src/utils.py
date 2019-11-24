import os
import os.path

import matplotlib.pyplot as plt
import seaborn as sns


def plot_matrix(matrix, plot_name):
    """Plot matrix to a file as a heat map.

    Parameters
    ----------
    matrix : array-like
        Data to visualize as a heat map.
    plot_name : str
        Plot's title as well as name of the output file.
    """
    if len(matrix.shape) != 2:
        raise ValueError('Only 2-D arrays can be displayed as a heat map.')
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Data should have a square shape.')

    sns.set(font_scale=1.25)
    fig, ax = plt.subplots(figsize=(matrix.shape[1], matrix.shape[0]))
    sns.heatmap(
        matrix, square=True,
        annot=True, fmt=".1f", linewidths=0.5, linecolor='black', ax=ax
    )
    plt.title(plot_name, fontsize=2*matrix.shape[0])

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            '..', 'graphics', plot_name + '.pdf'
        ),
        dpi=150, format='pdf'
    )
    plt.close(fig)

