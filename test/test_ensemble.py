"""Tests for the ensemble.py

"""

# Author          : Jin Kim jjinking(at)gmail(dot)com
# Creation date   : 2016.04.12
# Last Modified   : 2016.04.12
#
# License         : MIT

import numpy as np
import pandas as pd
import unittest

from datsci import ensemble


class TestEnsemble(unittest.TestCase):

    def test_weighted_avg(self):
        y1, y2, y3 = [1, 2, 3], [4, 5, 6], [7, 8, 9]
        df = pd.DataFrame({'y1': y1, 'y2': y2, 'y3': y3})

        # Check unweighted mean
        self.assertTrue(np.array_equal(
            df.mean(axis=1).values,
            ensemble.weighted_avg([y1, y2, y3])
        ))

        # Check weighted mean
        self.assertTrue(np.array_equal(
            ensemble.weighted_avg([y1, y2, y3], weights=[1, 1, 2]),
            np.array([19, 23, 27])
        ))


if __name__ == '__main__':
    unittest.main()
