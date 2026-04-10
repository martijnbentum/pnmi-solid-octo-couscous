import unittest

import numpy as np

from pnmi import build_joint_labels
from pnmi import cluster_purity
from pnmi import entropy
from pnmi import analyze_all_dummy_datasets
from pnmi import evaluate_labels
from pnmi import evaluate_streams
from pnmi import joint_distribution
from pnmi import marginals
from pnmi import mutual_information
from pnmi import phone_purity
from pnmi import perfect_pnmi_data
from pnmi import pnmi
from pnmi import select_codebook_streams


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
            dtype=float)
        np.testing.assert_allclose(joint, expected)

    def test_marginals_and_entropy_are_consistent(self):
        phone_labels = [0, 0, 1, 1]
        cluster_labels = [1, 1, 0, 0]

        phone_marginal, cluster_marginal = marginals(phone_labels,
            cluster_labels)

        np.testing.assert_allclose(phone_marginal, [0.5, 0.5])
        np.testing.assert_allclose(cluster_marginal, [0.5, 0.5])
        self.assertAlmostEqual(entropy(phone_marginal), np.log(2.0))

    def test_perfect_clustering_has_pnmi_one(self):
        phone_labels = [0, 0, 1, 1, 2, 2]
        cluster_labels = [4, 4, 7, 7, 9, 9]

        self.assertAlmostEqual(
            mutual_information(phone_labels, cluster_labels),
            entropy(np.array([2 / 6, 2 / 6, 2 / 6], dtype=float)))
        self.assertAlmostEqual(pnmi(phone_labels, cluster_labels), 1.0)
        self.assertAlmostEqual(phone_purity(phone_labels, cluster_labels), 1.0)
        self.assertAlmostEqual(cluster_purity(phone_labels, cluster_labels), 1.0)

    def test_independent_assignments_have_zero_information(self):
        phone_labels = [0, 0, 1, 1]
        cluster_labels = [0, 1, 0, 1]

        self.assertAlmostEqual(
            mutual_information(phone_labels, cluster_labels),
            0.0)
        self.assertAlmostEqual(pnmi(phone_labels, cluster_labels), 0.0)
        self.assertAlmostEqual(phone_purity(phone_labels, cluster_labels), 0.5)
        self.assertAlmostEqual(cluster_purity(phone_labels, cluster_labels), 0.5)

    def test_single_phone_class_returns_zero_pnmi(self):
        phone_labels = ['aa', 'aa', 'aa']
        cluster_labels = [0, 1, 1]

        self.assertAlmostEqual(
            entropy(marginals(phone_labels, cluster_labels)[0]),
            0.0)
        self.assertAlmostEqual(pnmi(phone_labels, cluster_labels), 0.0)
        self.assertAlmostEqual(phone_purity(phone_labels, cluster_labels), 1.0)
        self.assertAlmostEqual(cluster_purity(phone_labels, cluster_labels), 2 / 3)

    def test_evaluate_labels_returns_summary(self):
        result = evaluate_labels([0, 0, 1, 1], [0, 0, 1, 1])

        self.assertEqual(result['valid_frame_count'], 4)
        self.assertAlmostEqual(result['pnmi'], 1.0)
        self.assertAlmostEqual(result['phone_purity'], 1.0)
        self.assertAlmostEqual(result['cluster_purity'], 1.0)

    def test_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            joint_distribution([], [])

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            joint_distribution([0, 1], [0])


class SpidrRegressionTests(unittest.TestCase):
    def test_select_codebook_streams_supports_layer_range(self):
        codebooks = {
            5: [0, 1, 0, 1],
            6: [1, 1, 0, 0],
            7: [2, 2, 3, 3],
        }

        selected = select_codebook_streams(codebooks, start_layer=6,
            end_layer=7)

        self.assertEqual(list(selected.keys()), [6, 7])

    def test_build_joint_labels_combines_streams_per_frame(self):
        codebooks = {
            0: [0, 1, 0],
            1: [3, 3, 4],
        }

        joint = build_joint_labels(codebooks)

        self.assertEqual(joint.tolist(), [(0, 3), (1, 3), (0, 4)])

    def test_per_stream_evaluation_returns_one_result_per_stream(self):
        phone_labels = [0, 0, 1, 1, 2, 2, 3, 3]
        codebooks = {
            5: [0, 0, 0, 0, 1, 1, 1, 1],
            6: [0, 0, 1, 1, 0, 0, 1, 1],
        }

        results = evaluate_streams(phone_labels, codebooks,
            mode='per_stream')

        self.assertEqual(set(results.keys()), {5, 6})
        self.assertLess(results[5]['pnmi'], 1.0)
        self.assertLess(results[6]['pnmi'], 1.0)

    def test_joint_token_captures_complementary_streams(self):
        phone_labels = [0, 0, 1, 1, 2, 2, 3, 3]
        codebooks = {
            5: [0, 0, 0, 0, 1, 1, 1, 1],
            6: [0, 0, 1, 1, 0, 0, 1, 1],
        }

        per_stream = evaluate_streams(phone_labels, codebooks,
            mode='per_stream')
        joint = evaluate_streams(phone_labels, codebooks,
            mode='joint_token')

        self.assertLess(per_stream[5]['pnmi'], 1.0)
        self.assertLess(per_stream[6]['pnmi'], 1.0)
        self.assertAlmostEqual(joint['pnmi'], 1.0)

    def test_pooled_summary_computes_weighted_mean(self):
        phone_labels = [0, 0, 1, 1, 2, 2]
        codebooks = {
            0: [0, 0, 1, 1, 2, 2],
            1: [0, 1, 0, 1, -1, -1],
        }

        summary = evaluate_streams(phone_labels, codebooks,
            mode='pooled_summary', invalid_label=-1,
            return_diagnostics=True)

        self.assertAlmostEqual(summary['pnmi']['mean'], 0.5)
        self.assertAlmostEqual(summary['pnmi']['weighted_mean'], 0.6)
        self.assertEqual(summary['valid_frame_count_total'], 10)
        self.assertIn('mutual_information', summary)


class DummyDataRegressionTests(unittest.TestCase):
    def test_dummy_dataset_returns_direct_label_sequences(self):
        phone_labels, cluster_labels = perfect_pnmi_data()

        self.assertEqual(phone_labels.shape, (80,))
        self.assertEqual(cluster_labels.shape, (80,))
        self.assertEqual(phone_labels[:4].tolist(), ['aa', 'aa', 'aa', 'aa'])
        self.assertEqual(cluster_labels[:4].tolist(), ['c0', 'c0', 'c0', 'c0'])

    def test_dummy_datasets_have_expected_ordering(self):
        results = analyze_all_dummy_datasets()
        pnmis = {name: result['pnmi'] for name, result in results.items()}

        self.assertAlmostEqual(pnmis['perfect'], 1.0)
        self.assertGreater(pnmis['high'], pnmis['medium'])
        self.assertGreater(pnmis['medium'], pnmis['low'])
        self.assertGreater(pnmis['low'], pnmis['none'])
        self.assertAlmostEqual(pnmis['none'], 0.0)


if __name__ == '__main__':
    unittest.main()
