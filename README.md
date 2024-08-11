# Adobe_gensolve

---

# Doodle Regularization and Symmetry Check

This script is designed to regularize and smoothen doodles, comparing them with regular shapes to check for symmetry.

## Table of Contents

- [Purpose](#purpose)
- [Input](#input)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Output](#output)
- [Authors](#authors)

## Purpose

The purpose of this script is to regularize and smoothen doodles to compare them with regular shapes and assess the symmetry of these doodles.

## Input

The script requires a CSV file as input, which contains the doodle data.

## Dependencies

Ensure the following libraries are installed before running the script:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

You can install the required libraries using pip:

```sh
pip install numpy matplotlib scipy scikit-learn
```

## Usage

To use the script, run the following command:

```sh
python Submission.py <path_to_csv_file>
```

Replace `<path_to_csv_file>` with the path to your CSV file containing the doodle data.

## Output

The script will process the input doodle, regularize it, smooth it, and check its symmetry. The results will be visualized and can be analyzed for further use.

## Authors

- Thakkar Prince
- Jyoti Gupta

---
