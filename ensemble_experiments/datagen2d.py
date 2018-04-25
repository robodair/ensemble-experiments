"""
Utilities for generating and visualising 2D datasets

Discriminator of these datasets is fixed to 'y = 0.2 * (1 + cos(7 * pi * x)) + 0.65 * x^2'
"""

import sys

import csv
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_A = 0
CLASS_B = 1

def discriminator(x):
    """
    Discriminator of y = 0.1 * ( 2.5 + cos( 3 * pi * x) ) + 0.5 * x^2
    """
    return 0.1 * ( 2.5 + np.cos( 3 * np.pi * x) ) + 0.5 * np.power(x, 2)


def generate_uniform_data():
    """
    Generate a uniform dataset with points every 0.01 spaces
    Used when plitting to visualise the discriminator
    """
    data = pd.DataFrame()
    x = np.array([])
    y = np.array([])

    for xval in np.arange(0, 1, 1/100):
        for yval in np.arange(0, 1, 1/100):
            x = np.append(x, xval)
            y = np.append(y, yval)
    data['x'] = x
    data['y'] = y
    comp = discriminator(data['x'])
    data['realclass'] = None
    data.loc[data['y'] > comp, 'realclass'] = CLASS_A
    data.loc[data['y'] < comp, 'realclass'] = CLASS_B

    return data


def generate_data(count: int, error: int, seed: int = None) -> pd.DataFrame:
    """
    Generate a 2D dataset with two classes.

    Arguments:
        count: The number of data points
        error: Percentage of points to incorrectly classify
        seed: Random seed to use for the data

    Returns:
        A pandas dataframe with columns for x, y, class, and real class
    """
    if seed is not None:
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


