import numpy as np

from pnmi import cluster_hidden_states
from pnmi import evaluate_streams
from pnmi import joint_distribution
from pnmi import pnmi


def hubert_style_example():
    phone_labels = ['aa', 'aa', 'b', 'b', 'b', 'k']
    cluster_labels = [0, 0, 1, 1, 2, 2]

    print('HuBERT-style single stream')
    print(joint_distribution(phone_labels, cluster_labels))
    print('pnmi:', pnmi(phone_labels, cluster_labels))
    print()


def spidr_style_example():
    phone_labels = [0, 0, 1, 1, 2, 2, 3, 3]
    codebooks = {
        5: [0, 0, 0, 0, 1, 1, 1, 1],
        6: [0, 0, 1, 1, 0, 0, 1, 1],
        7: [0, 1, 0, 1, 0, 1, 0, 1],
    }

    print('SpidR-style multi-stream')
    print('per stream:', evaluate_streams(phone_labels, codebooks,
        'per_stream'))
    print('joint token:', evaluate_streams(phone_labels, codebooks,
        'joint_token'))
    print(
        'pooled summary:',
        evaluate_streams(phone_labels, codebooks, 'pooled_summary'),
    )
    print()


def clustering_example():
    phone_labels = [0, 0, 1, 1, 2, 2]
    hidden_states = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [4.0, 4.0],
            [4.1, 3.9],
            [8.0, 8.0],
            [8.2, 8.1],
        ],
        dtype = float,
    )

    try:
        cluster_labels = cluster_hidden_states(hidden_states, n_clusters=3)
    except ImportError:
        print('Install scikit-learn to run the clustering example.')
        return

    print('Cluster hidden states then compute PNMI')
    print('cluster labels:', cluster_labels.tolist())
    print('pnmi:', pnmi(phone_labels, cluster_labels))


if __name__ == '__main__':
    hubert_style_example()
    spidr_style_example()
    clustering_example()
