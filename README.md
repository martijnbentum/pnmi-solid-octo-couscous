# pnmi

Small NumPy-only utilities for computing phone-normalized mutual
information (PNMI) and related clustering metrics on frame-level phone
labels and frame-level cluster labels, following the HuBERT paper.

Reference: HuBERT, Wei-Ning Hsu et al., arXiv:2106.07447.

## Installation

```bash
pip install git+https://git@github.com/martijnbentum/pnmi-solid-octo-couscous.git
```

## Example Usage

```python
from pnmi import cluster_purity
from pnmi import joint_distribution
from pnmi import phone_purity
from pnmi import pnmi

phone_labels = ['aa', 'aa', 'b', 'b', 'b', 'k']
cluster_labels = [0, 0, 1, 1, 2, 2]

joint = joint_distribution(phone_labels, cluster_labels)

print(joint)
print('pnmi:', pnmi(phone_labels, cluster_labels))
print('phone purity:', phone_purity(phone_labels, cluster_labels))
print('cluster purity:', cluster_purity(phone_labels, cluster_labels))
```
