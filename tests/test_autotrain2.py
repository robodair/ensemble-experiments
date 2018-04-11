import unittest
from pathlib import Path
import pandas
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from ensemble_experiments import autotrain2, datagen2d

import pytest

class TestAutoTrain(unittest.TestCase):

    @pytest.mark.smoke
    def test_autotrain_smoke(self):
        """
        Very simply test autotrain with tiny generated datasets
        """
        data = datagen2d.generate_data(300, 10, 1)
        train_data = data[::2]
        test_data = data[1::2]
        op_model, ot_model = autotrain2.autotrain(
            train_data,
            test_data,
            Path(__file__).parent,
            min_nodes=2,
            max_nodes=9,
            max_epochs=2000,
            verbose=0
        )

