# pnmi

Utilities for computing phone-normalized mutual information (PNMI) and
related statistics on frame-level phone labels and frame-level discrete
unit labels.

The package supports:
- HuBERT-style evaluation with one discrete label per frame
- wav2vec2 or hidden-state clustering pipelines
- SpidR-style evaluation with one or more codebook streams

References:
- HuBERT, Wei-Ning Hsu et al., arXiv:2106.07447
- SpidR, Maxime Poli et al., TMLR 2025 / OpenReview:
  https://openreview.net/forum?id=E7XAFBpfZs
- SpidR arXiv version: https://arxiv.org/abs/2512.20308

## Installation

```bash
pip install git+https://git@github.com/martijnbentum/pnmi-solid-octo-couscous.git
```

For local development:

```bash
./scripts/install_local.sh
```

This keeps local setuptools metadata and build artifacts under `.build/`,
including moving any root-level `pnmi.egg-info` created by editable install
back into `.build/`.

## Core API

- `evaluate_labels(phone_labels, cluster_labels)` for one frame-level stream
- `pnmi(phone_labels, cluster_labels)` for the scalar HuBERT-style metric
- `evaluate_streams(phone_labels, codebooks, mode=...)` for SpidR-style
  multi-stream evaluation
- `cluster_hidden_states(hidden_states, n_clusters)` for clustering-based
  workflows when scikit-learn is installed

## Workflow: HuBERT

For a HuBERT-style setup you typically already have:
- frame-level phone labels from a forced aligner or annotated corpus
- one discrete label per frame, for example from k-means over hidden states

```python
from pnmi import evaluate_labels

phone_labels = ['sil', 'sil', 'aa', 'aa', 't', 't', 'iy', 'iy']
cluster_labels = [3, 3, 7, 7, 2, 2, 9, 9]

result = evaluate_labels(phone_labels, cluster_labels,
    return_diagnostics=True)

print('pnmi:', result['pnmi'])
print('phone purity:', result['phone_purity'])
print('cluster purity:', result['cluster_purity'])
print('mutual information:', result['mutual_information'])
```

If you need to cluster frame-level hidden states first:

```python
import numpy as np

from pnmi import cluster_hidden_states
from pnmi import evaluate_labels

hidden_states = np.array([
    [0.0, 0.1],
    [0.1, 0.0],
    [4.0, 4.1],
    [4.2, 3.9],
    [8.0, 8.1],
    [8.2, 7.9],
], dtype=float)
phone_labels = ['aa', 'aa', 't', 't', 'iy', 'iy']

cluster_labels = cluster_hidden_states(hidden_states, n_clusters=3)
result = evaluate_labels(phone_labels, cluster_labels)
print(result['pnmi'])
```

## Workflow: SpidR

For a SpidR-style setup you typically have:
- frame-level phone labels
- one or more codebook streams predicted by the model
- optional layer selection when multiple codebooks are available

```python
from pnmi import evaluate_streams

phone_labels = [0, 0, 1, 1, 2, 2, 3, 3]
codebooks = {
    5: [0, 0, 0, 0, 1, 1, 1, 1],
    6: [0, 0, 1, 1, 0, 0, 1, 1],
    7: [0, 1, 0, 1, 0, 1, 0, 1],
}

per_stream = evaluate_streams(phone_labels, codebooks,
    mode='per_stream', start_layer=5, end_layer=7)
joint = evaluate_streams(phone_labels, codebooks,
    mode='joint_token', layers=[5, 6, 7])
summary = evaluate_streams(phone_labels, codebooks,
    mode='pooled_summary', layers=[5, 6, 7],
    return_diagnostics=True)

print('stream 5 pnmi:', per_stream[5]['pnmi'])
print('joint-token pnmi:', joint['pnmi'])
print('pooled weighted mean:', summary['pnmi']['weighted_mean'])
```

The intended comparison pattern for SpidR-like systems is:
- inspect each stream separately with `mode='per_stream'`
- report the combined symbol view with `mode='joint_token'`
- use `mode='pooled_summary'` for compact cross-stream summaries

## Examples

- [examples/example_usage.py](/Users/martijn.bentum/repos/pnmi/examples/example_usage.py)
  contains HuBERT-style, SpidR-style, and clustering examples
- [examples/dummy_data_analysis.py](/Users/martijn.bentum/repos/pnmi/examples/dummy_data_analysis.py)
  reports `perfect`, `high`, `medium`, `low`, and `none` dummy conditions
