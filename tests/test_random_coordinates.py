import unittest

import numpy as np

import sklearn_supp.random_coordinates as random_coordinates


class TestRandomCoordinateForestClassifier(unittest.TestCase):
    """These are just some simple sanity checks to make sure we don't get
    exceptions.
    """

    def test_simple(self):
        X = [[0], [1]]
        y = [0, 1]
        classifier = random_coordinates.RandomCoordinateForestClassifier(
            n_estimators=50)
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        print(y_pred)
        eq = np.all(y_pred == y)
        self.assertTrue(eq)

    def test_transform_dimension(self):
        X = [[0, 0], [1, 1]]
        X = np.array(X)
        y = [0, 1]
        classifier = random_coordinates.RandomCoordinateForestClassifier(
            n_estimators=50, transform_dimension=2)
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        print(y_pred)
        eq = np.all(y_pred == y)
        self.assertTrue(eq)
