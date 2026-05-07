# Machine Learning-Based Phishing Website Detection

**Comparative Evaluation and Feature Importance Analysis of Machine Learning Models for Phishing Website Detection**

**Authors:** Murshud Azimov, Javid Shukurov  
**Affiliation:** Western Caspian University

## Overview

This repository contains the code, results, and paper for an undergraduate research project comparing machine learning models for phishing website detection using the UCI Phishing Websites Dataset.

📄 **[Read the Paper (PDF)](phishing_detection_ml.pdf)**

## Research Questions

- **RQ1:** Which machine learning model performs best for phishing website detection?
- **RQ2:** Which evaluation metrics are most informative for phishing detection, especially considering false negatives?
- **RQ3:** Which website features contribute most to phishing classification?

## Key Results

| Model | F1-score | ROC-AUC |
|-------|----------|---------|
| **Random Forest** | **0.9733** | **0.9967** |
| Decision Tree | 0.9650 | 0.9762 |
| SVM-RBF | 0.9507 | 0.9901 |
| Gradient Boosting | 0.9435 | 0.9913 |
| KNN | 0.9397 | 0.9856 |
| Logistic Regression | 0.9224 | 0.9786 |

**Top 3 Most Important Features:**
1. `sslfinal_state` (SSL certificate status)
2. `url_of_anchor` (anchor URL patterns)
3. `prefix_suffix` (hyphen usage in domain)

## Dataset

UCI Phishing Websites Dataset (ID: 327)
- 11,055 instances
- 30 input features
- Binary classification: Phishing vs Legitimate
- Source: https://archive.ics.uci.edu/dataset/327/phishing+websites

## Models

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM-RBF)
- Decision Tree
- Random Forest
- Gradient Boosting

## Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Project Structure

```
phishing-detection-ml/
├── data/
│   └── README.md
├── notebooks/
│   └── phishing_detection_experiments.ipynb
├── src/
│   └── train_models.py
├── paper/
│   ├── main.tex
│   ├── phishing_detection_paper.md
│   └── paper_outline.md
├── results/
│   └── (generated after running experiments)
├── phishing_detection_ml.pdf
├── README.md
├── requirements.txt
└── LICENSE
```

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the full experiment:
```bash
python src/train_models.py
```

3. Or use the Jupyter notebook for interactive exploration:
```bash
jupyter notebook notebooks/phishing_detection_experiments.ipynb
```

## Results

Results will be saved in the `results/` directory after running the experiments:
- `test_results.csv` - Test set performance metrics
- `cv_results.csv` - Cross-validation results
- `feature_importance.csv` - Permutation feature importance
- `top_k_feature_results.csv` - Feature subset analysis
- `confusion_matrix.png` - Confusion matrix visualization
- `feature_importance.png` - Top features bar chart

## Citation

If you use this work, please cite:

```
Azimov, M., & Shukurov, J. (2026). Comparative Evaluation and Feature Importance 
Analysis of Machine Learning Models for Phishing Website Detection. 
Western Caspian University.
```

## License

MIT License
