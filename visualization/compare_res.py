# Libraries
import random
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR = ['#B6BFF2', '#04C4D9', '#F2C12E', '#F26363', '#BF7E04', '#7F2F56', '#E8B9B5', '#63CAF3', '#F27405', '#68BD44']
MARKER = ['D', '^', 'o', 'H', '+', 'x', 's', 'p', '*', '3']


def cross_methods_plot(matrics_name, method_name, values, save_name, postfix='.jpg', color=COLOR,
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


def cross_tasks_plot(matrics_name, method_name, n_task, save_name, *values, error_bar=False, postfix='.jpg',
                     color=COLOR, markers=MARKER):
    '''
    Comparison of experimental results among tasks
    :param matrics_name: selected performance matrics, e.g. ['BWT','ACC', 'FWT']
    :param method_name: selected CL method applied on the dataset, e.g. ['SI','EWC']
    :param n_task: number of tasks for the experiment
    :param save_name: name of the img to be saved
    :param values: number of values = number of performance matrices, each with 2D/3D array w/ rows indicating methods, columns indicating tasks
    :param error_bar: whether to present error bar in the plot, default is False
    :param postfix: type of the img, e.g. jpg, png, pdf, etc.
    :param color: alternative colors
    :param markers: alternative markers
    :return: save figure
    '''
    if error_bar:
        assert len(np.array(values[0]).shape) == 3
        assert len(method_name) == np.array(values[0]).shape[1]
    else:
        assert len(np.array(values[0]).shape) == 2
        assert len(method_name) == np.array(values[0]).shape[0]

    assert len(matrics_name) == len(values)

    for i in range(len(matrics_name)):
        values_i = np.array(values[i])

        plt.figure()
        x = np.arange(1, n_task + 1)
        for j in range(len(method_name)):
            if error_bar:
                assert len(values_i.shape) == 3
                y = values_i[0, :, :][j]
                error = values_i[1, :, :][j]
                plt.plot(x, y, color=color[j], marker=markers[j], label=method_name[j])
                plt.fill_between(x, y - error, y + error, alpha=0.2)
            else:
                assert len(values_i.shape) == 2
                y = values_i[j]
                plt.plot(x, y, color=color[j], marker=markers[j], label=method_name[j])
        plt.xlabel('Encountered Batches')
        plt.ylabel(matrics_name[i])
        plt.xticks(x)
        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.08), ncol=len(method_name))
        plt.grid()
        plt.savefig(save_name + '_' + matrics_name[i] + postfix)


if __name__ == '__main__':
    '''
    # test for cross_methods_plot
    matrics_name = ['BWT', 'ACC', 'FWT', 'Overall Accuracy']
    method_name = ['SI', 'EWC', 'Naive', 'LwF']
    values = [[38, 1.5, 30, 4], [29, 10, 9, 34], [8, 39, 23, 24], [7, 31, 33, 14], [28, 15, 32, 14]]
    save_name = 'test'
    spider = True
    bar = True
    cross_methods_plot(matrics_name, method_name, values, save_name, spider=True, bar=True)
    '''

    matrics_name = ['ACC', 'FWT']
    method_name = ['SI', 'EWC', 'Naive']
    n_task = 5
    save_name = 'test'
    # w/ error
    values_error_1 = [[[0.9410, 0.4421, 0.2102, 0.2931, 0.5084], [0.8520, 0.9633, 0.3652, 0.5632, 0.3695],
                       [0.6985, 0.4052, 0.8741, 0.9537, 0.6147]],
                      [[0.0159, 0.0121, 0.0104, 0.0121, 0.0115], [0.0149, 0.0101, 0.0113, 0.0125, 0.0106],
                       [0.0164, 0.0117, 0.0124, 0.0196, 0.0143]]]

    values_error_2 = np.array([[[0.6507, 0.5888, 0.0784, 0.4622, 0.3905], [0.9873, 0.3803, 0.3297, 0.3855, 0.6573],
                                [0.8326, 0.1652, 0.0235, 0.7906, 0.8511]],
                               [[0.0159, 0.0121, 0.0104, 0.0121, 0.0115], [0.0149, 0.0101, 0.0113, 0.0125, 0.0106],
                                [0.0164, 0.0117, 0.0124, 0.0196, 0.0143]]])
    # w/o error
    values_1 = np.array([[0.9410, 0.4421, 0.2102, 0.2931, 0.5084], [0.8520, 0.9633, 0.3652, 0.5632, 0.3695],
                         [0.6985, 0.4052, 0.8741, 0.9537, 0.6147]])
    values_2 = np.array([[0.6507, 0.5888, 0.0784, 0.4622, 0.3905], [0.9873, 0.3803, 0.3297, 0.3855, 0.6573],
                         [0.8326, 0.1652, 0.0235, 0.7906, 0.8511]])

    cross_tasks_plot(matrics_name, method_name, n_task, save_name, values_error_1, values_error_2, error_bar=True)
