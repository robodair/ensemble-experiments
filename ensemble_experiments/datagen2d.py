"""
Utilities for generating and visualising 2D datasets

Discriminator of these datasets is fixed to 'y = 0.2 * (1 + cos(7 * pi * x)) + 0.65 * x^2'
"""


import sys
import argparse
import csv
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as pyplot

CLASS_A = 0
CLASS_B = 1

def make_plot(data):
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

    pyplot.show()


def discriminator(x):
    """
    Discriminator of y = 0.2 * (1 + cos(7 * pi * x)) + 0.65 * x^2
    """
    return 0.2 * (1 + np.cos(7 * np.pi * x)) + 0.65 * np.power(x, 2)


def generate_data(count: int, error: int, seed: int) -> pd.DataFrame:
    """
    Generate a 2D dataset with two classes.

    Arguments:
        count: The number of data points
        error: Percentage of points to incorrectly classify
        seed: Random seed to use for the data

    Returns:
        A pandas dataframe with columns for x, y, class, and real class
    """
    np.random.seed(seed)
    data = pd.DataFrame()

    data['x'] = np.random.random(count)
    data['y'] = np.random.random(count)

    comp = discriminator(data['x'])
    data['class'] = None
    data.loc[data['y'] > comp, 'class'] = CLASS_A
    data.loc[data['y'] < comp, 'class'] = CLASS_B
    data['realclass'] = data['class']

    # Extract view of points that are to be an incorrect class
    last_incorrect_point = count * error // 100

    fake_a_classes_mask = (data[:last_incorrect_point]['class'] == CLASS_B).values
    fake_b_classes_mask = (data[:last_incorrect_point]['class'] == CLASS_A).values

    # Swap the classes of the incorrect classification
    data.loc[fake_a_classes_mask, 'class'] = CLASS_A
    data.loc[fake_b_classes_mask, 'class'] = CLASS_B

    return data


def main(args):
    data = generate_data(args.count, args.error, args.seed)

    if args.plot:
        make_plot(data)

    if args.save_file:
        data.to_csv(args.save_file)


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument("count",
                        type=int,
                        help="Integer, Number of datapoints to produce")
    parser.add_argument("error",
                        type=int,
                        help="Integer, Percentage of data points to incorrectly classify")
    parser.add_argument("--plot",
                        default=False,
                        action="store_true",
                        help="Visualize the dataset by plotting it")
    parser.add_argument("--seed", type=int,
                        default=2018,
                        help="Random seed to use for Numpy")
    parser.add_argument("--save-file",
                        default=None,
                        type=Path,
                        help="Save data to specified file in csv format")
    parser.set_defaults(func=main)
