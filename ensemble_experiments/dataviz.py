"""
Visualise a generated dataset for the experiment by plotting it, it's discriminator and errored
points
"""

import argparse
from pathlib import Path

import numpy as np

import matplotlib.pyplot as pyplot

from ensemble_experiments.datagen2d import generate_data, discriminator, CLASS_A, CLASS_B

def plot_dataset(data):
    # Plot Discriminator
    t = np.arange(0, 1, 0.01)
    s = discriminator(t)
    pyplot.plot(t, s)

    # Class A points
    pyplot.scatter(
        data.loc[data['class'] == CLASS_A, 'x'],
        data.loc[data['class'] == CLASS_A, 'y'],
        c='red',
        marker='.',
        label='Class A'
    )

    # Class B points
    pyplot.scatter(
        data.loc[data['class'] == CLASS_B, 'x'],
        data.loc[data['class'] == CLASS_B, 'y'],
        c='green',
        marker='.',
        label='Class B'
    )

    # Incorrectly labelled points
    pyplot.scatter(
        data.loc[data['class'] != data['realclass'], 'x'],
        data.loc[data['class'] != data['realclass'], 'y'],
        c='purple',
        marker='.',
        label='Incorrect Class'
    )

def main(args):
    data = generate_data(args.count, args.error, args.seed)

    plot_dataset(data)

    if args.save_file:
        data.to_csv(args.save_file)

    pyplot.legend()
    pyplot.show() # Blocking


def setup_parser(parser: argparse.ArgumentParser):
    parser.add_argument('count',
                        type=int,
                        help='Integer, Number of datapoints to produce')
    parser.add_argument('error',
                        type=int,
                        help='Integer, Percentage of data points to incorrectly classify')
    parser.add_argument('--seed', type=int,
                        default=2018,
                        help='Random seed to use for Numpy')
    parser.add_argument('--save-file',
                        default=None,
                        type=Path,
                        help='Save data to specified file in csv format')
    parser.set_defaults(func=main)
