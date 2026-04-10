# pnmi

Small utilities for computing phone-normalized mutual information
(PNMI) and related clustering metrics on frame-level phone labels and
frame-level discrete unit labels.

The package supports both:
- HuBERT-style single frame-level label streams
- SpidR-style multi-codebook or multi-stream discrete predictions

References:
- HuBERT, Wei-Ning Hsu et al., arXiv:2106.07447
- SpidR, Maxime Poli et al., TMLR 2025 / OpenReview:
  https://openreview.net/forum?id=E7XAFBpfZs
- arXiv version of SpidR: https://arxiv.org/abs/2512.20308

## Installation

```bash
pip install git+https://git@github.com/martijnbentum/pnmi-solid-octo-couscous.git
```

## Example Usage

```python
from pnmi import evaluate_streams
from pnmi import pnmi

# HuBERT-style single label stream
phone_labels = ['aa', 'aa', 'b', 'b', 'b', 'k']
cluster_labels = [0, 0, 1, 1, 2, 2]
print(pnmi(phone_labels, cluster_labels))

# SpidR-style codebooks from one or more layers
codebooks = {
    5: [0, 0, 0, 0, 1, 1, 1, 1],
    6: [0, 0, 1, 1, 0, 0, 1, 1],
}
print(evaluate_streams(phone_labels = [0, 0, 1, 1, 2, 2, 3, 3],
    codebooks = codebooks, mode = 'per_stream'))
print(evaluate_streams(phone_labels = [0, 0, 1, 1, 2, 2, 3, 3],
    codebooks = codebooks, mode = 'joint_token'))
```

See [examples/example_usage.py](/Users/martijn.bentum/repos/pnmi/examples/example_usage.py)
for single-stream, multi-stream, and clustering-based usage.
