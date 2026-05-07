# Dataset

This project uses the **UCI Phishing Websites Dataset** (ID: 327).

## Source

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/327/phishing+websites
- The dataset is fetched programmatically using the `ucimlrepo` Python package.

## Description

- **Instances:** 11,055
- **Features:** 30 (all integer-encoded)
- **Target:** Binary classification
  - Phishing (-1 in original, mapped to 1)
  - Legitimate (1 in original, mapped to 0)

## Citation

Please cite the dataset as specified on the UCI repository page when using it in publications.

## Note

The dataset is NOT stored in this repository. It is downloaded automatically when running the experiments via the `ucimlrepo` package.
