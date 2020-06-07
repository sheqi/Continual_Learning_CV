# Libraries
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import random
import pandas as pd


def cross_methods_plot(matrics_name, method_name, values, save_name, postfix='.jpg',
                       color=['#B6BFF2', '#04C4D9', '#F2C12E', '#F26363', '#BF7E04', '#7F2F56', '#E8B9B5', '#63CAF3',
                              '#F27405', '#68BD44'],
                       spider=True, bar=False):
    '''
     Comparison of experimental results among methods
    :param matrics_name: selected performance matrics, e.g. ['BWT','ACC', 'FWT']
    :param method_name: selected CL method applied on the dataset, e.g. ['SI','EWC']
    :param df: 2D array w/ rows indicating matrices, columns indicating methods
    :param save_name: name of the img to be saved
    :param postfix: type of the img, e.g. jpg, png, pdf, etc.
    :param color: alternative colors
    :param spider: use spider plot, default is True
    :param bar: use bar plot, default is False
    :return: save figure
    '''

    if not spider and not bar:
        raise NotImplementedError("No figure type is selected.")

    raw_data = {'matrices\method': matrics_name}
    for i in range(len(method_name)):
        raw_data[method_name[i]] = values[i]

    df_col = ['matrices\method'] + method_name
    df = pd.DataFrame(raw_data, columns=df_col)

    # number of matrics
    N = len(matrics_name)

    assert N == df.shape[0]

    if spider:
        # the angle of each axis in the plot: divide the plot / number of variable
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)

        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], method_name)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
        plt.ylim(0, 40)

        color_select = random.sample(color, len(matrics_name))
        for i in range(len(matrics_name)):
            values = df.loc[i].drop('matrices\method').values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=matrics_name[i], color=color_select[i])

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.savefig(save_name + '_spider' + postfix)

    if bar:
        ax = df.plot.bar(rot=0, color=color[:len(method_name)], width=0.8)
        '''
        # show values
        for p in ax.patches[1:]:
            h = p.get_height()
            x = p.get_x() + p.get_width() / 2.
            if h != 0:
                ax.annotate("%g" % p.get_height(), xy=(x, h), xytext=(0, 4), rotation=90,
                            textcoords="offset points", ha="center", va="bottom")
        '''
        ax.set_xlim(-0.5, None)
        ax.margins(y=0.05)
        ax.legend(ncol=len(df.columns), loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.08),
                  borderaxespad=0, mode="expand")
        ax.set_xticklabels(df["matrices\method"])
        ax.grid()
        plt.savefig(save_name + '_bar' + postfix)


if __name__ == '__main__':
    matrics_name = ['BWT', 'ACC', 'FWT', 'Overall Accuracy']
    method_name = ['SI', 'EWC', 'Naive', 'LwF']
    values = [[38, 1.5, 30, 4], [29, 10, 9, 34], [8, 39, 23, 24], [7, 31, 33, 14], [28, 15, 32, 14]]
    save_name = 'test'
    spider = False
    bar = False
    cross_methods_plot(matrics_name, method_name, values, save_name, spider=True, bar=True)
