import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as pyplot

import numpy as np

from ensemble_experiments.datagen2d import discriminator

CLASS_A = 0
CLASS_B = 1

def make_plot(data, show=True):
    """
    Create a plot of the data and display it
    """
    t = np.arange(0, 1, 0.01)
    s = discriminator(t)
    # Plot Discriminator
    pyplot.plot(t, s)

    # Class A points
    pyplot.scatter(
        data.loc[data['class'] == CLASS_A, 'x'],
        data.loc[data['class'] == CLASS_A, 'y'],
        c="red", marker='.'
    )

    # Class B points
    pyplot.scatter(
        data.loc[data['class'] == CLASS_B, 'x'],
        data.loc[data['class'] == CLASS_B, 'y'],
        c="green", marker='.'
    )

    # Incorrectly labelled points
    pyplot.scatter(
        data.loc[data['class'] != data['realclass'], 'x'],
        data.loc[data['class'] != data['realclass'], 'y'],
        c="purple", marker='.'
    )

    if show:
        pyplot.show()

def plot_discriminator_visualisation(vis_data, train_data):
    pyplot.clf()
    # Plot data
    # Class A points
    pyplot.scatter(
        vis_data.loc[vis_data['class'] == CLASS_A, 'x'],
        vis_data.loc[vis_data['class'] == CLASS_A, 'y'],
        c="pink", marker='.'
    )

    # Class B points
    pyplot.scatter(
        vis_data.loc[vis_data['class'] == CLASS_B, 'x'],
        vis_data.loc[vis_data['class'] == CLASS_B, 'y'],
        c="skyblue", marker='.'
    )


    pyplot.ion()
    pyplot.show()
    pyplot.draw()
    make_plot(train_data, show=False)
    pyplot.pause(0.1)
