import unittest
from pathlib import Path
import pandas

from ensemble_experiments import autotrain, datagen2d

class TestAutotrain(unittest.TestCase):

    def test_autotrain_smoke(self):
        """
        Very simply test autotrain with tiny generated datasets
        """
        data = datagen2d.generate_data(75, 10, 2018)
        test_data = datagen2d.generate_data(75, 10, 2019)
        network = autotrain.autotrain(data, test_data, min_nodes=2, max_nodes=2)

