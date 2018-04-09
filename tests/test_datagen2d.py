import unittest
import ensemble_experiments
import ensemble_experiments.datagen2d as d2d

class TestDataGen2d(unittest.TestCase):

    def test_generate_simple_dataset(self):
        data = d2d.generate_data(100, 20, 1)
        assert len(data) == 100
        assert (data['class'] == data['realclass']).sum() == 80

    def test_generate_odd_number_dataset(self):
        data = d2d.generate_data(101, 20, 1)
        assert len(data) == 101
        # error classes would have been floored to 20, 101 * 20 // 100
        assert (data['class'] == data['realclass']).sum() == 81
