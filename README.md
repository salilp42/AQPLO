# Adaptive Quick-sort Leave-Pair-Out (AQPLO) Cross-validation

This repository contains the implementation and experimental evaluation of Adaptive Quick-sort Leave-Pair-Out (AQPLO) cross-validation, a novel approach to cross-validation for machine learning models. 

AQPLO is inpsired by QPLO (as per the original Quicksort CV paper by Numminen et al : https://link.springer.com/article/10.1007/s00180-022-01288-3)

## Overview

AQPLO introduces several key improvements over traditional cross-validation methods:
- Efficient group-wise comparisons using adaptive quicksort
- Enhanced caching mechanism for performance optimization
- Memory-efficient implementation for large datasets
- Automatic adaptation to dataset characteristics

## Repository Structure

- `quicksort_cvs.py`: Core implementation of QPLO and AQPLO algorithms
  - `QPLO` class: Base implementation with enhanced caching 
  - `AQPLO` class: Advanced implementation with adaptive parameters
- `main.py`: Script to reproduce experimental results
  - Includes comparisons with GSKF, LOGO, and TLPO methods
  - Supports multiple datasets (Breast Cancer, Diabetes)

## Requirements

- Python 3.x
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- Jupyter >= 1.0.0 (for notebook execution)

## Installation

```bash
git clone https://github.com/salilp42/AQPLO-Cross-Validation.git
cd AQPLO-Cross-Validation
pip install -r requirements.txt
```

## Usage

The package provides two main classes: `QPLO` and `AQPLO`. Here's a basic example:

```python
from quicksort_cvs import QPLO, AQPLO
from sklearn.linear_model import LogisticRegression

# Initialize with your classifier and data
classifier = LogisticRegression()
qplo = QPLO(classifier, X, y, groups)
aqplo = AQPLO(classifier, X, y, groups)

# Get sorted groups
qplo_sorted = qplo.sort()
aqplo_sorted = aqplo.sort()
```

For detailed examples and experimental results:
1. Run `main.py` to reproduce the core experiments

## Experimental Results

Our experiments show that AQPLO achieves:
- Consistant AUC scores compared to validated, unbiased methods
- Faster execution times, especially on larger datasets
- Better memory efficiency through adaptive caching
- More robust performance across different dataset characteristics

## Citation

If you use this code in your research, please cite:
```
@article{patel2024aqplo,
  title={AQPLO: Adaptive Quick-sort Leave-Pair-Out Cross-validation},
  author={Patel, Salil},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
