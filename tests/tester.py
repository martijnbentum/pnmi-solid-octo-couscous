import unittest

import numpy as np

from pnmi import cluster_purity
from pnmi import entropy
from pnmi import joint_distribution
from pnmi import marginals
from pnmi import mutual_information
from pnmi import phone_purity
from pnmi import pnmi


class PnmiRegressionTests(unittest.TestCase):
    def test_joint_distribution_matches_expected_counts(self):
        phone_labels = ['aa', 'aa', 'b', 'b', 'b', 'k']
        cluster_labels = [0, 0, 1, 1, 2, 2]

        joint = joint_distribution(phone_labels, cluster_labels)

        expected = np.array(
            [
                [2 / 6, 0 / 6, 0 / 6],
                [0 / 6, 2 / 6, 1 / 6],
                [0 / 6, 0 / 6, 1 / 6],
            ],
            dtype = float,
        )
        np.testing.assert_allclose(joint, expected)

    def test_marginals_and_entropy_are_consistent(self):
        phone_labels = [0, 0, 1, 1]
        cluster_labels = [1, 1, 0, 0]

        phone_marginal, cluster_marginal = marginals(
            phone_labels,
            cluster_labels,
        )

        np.testing.assert_allclose(phone_marginal, [0.5, 0.5])
        np.testing.assert_allclose(cluster_marginal, [0.5, 0.5])
        self.assertAlmostEqual(entropy(phone_marginal), np.log(2.0))

    def test_perfect_clustering_has_pnmi_one(self):
        phone_labels = [0, 0, 1, 1, 2, 2]
        cluster_labels = [4, 4, 7, 7, 9, 9]

        self.assertAlmostEqual(mutual_information(phone_labels, cluster_labels),
            entropy(np.array([2 / 6, 2 / 6, 2 / 6], dtype = float)))
        self.assertAlmostEqual(pnmi(phone_labels, cluster_labels), 1.0)
        self.assertAlmostEqual(phone_purity(phone_labels, cluster_labels), 1.0)
        self.assertAlmostEqual(cluster_purity(phone_labels, cluster_labels), 1.0)

    def test_independent_assignments_have_zero_information(self):
        phone_labels = [0, 0, 1, 1]
        cluster_labels = [0, 1, 0, 1]

        self.assertAlmostEqual(mutual_information(phone_labels, cluster_labels),
            0.0)
        self.assertAlmostEqual(pnmi(phone_labels, cluster_labels), 0.0)
        self.assertAlmostEqual(phone_purity(phone_labels, cluster_labels), 0.5)
        self.assertAlmostEqual(cluster_purity(phone_labels, cluster_labels), 0.5)

    def test_single_phone_class_returns_zero_pnmi(self):
        phone_labels = ['aa', 'aa', 'aa']
        cluster_labels = [0, 1, 1]

        self.assertAlmostEqual(entropy(marginals(phone_labels, cluster_labels)[0]),
            0.0)
        self.assertAlmostEqual(pnmi(phone_labels, cluster_labels), 0.0)
        self.assertAlmostEqual(phone_purity(phone_labels, cluster_labels), 1.0)
        self.assertAlmostEqual(cluster_purity(phone_labels, cluster_labels),
            2 / 3)

    def test_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            joint_distribution([], [])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            joint_distribution([0, 1], [0])


if __name__ == '__main__':
    unittest.main()
