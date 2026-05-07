# Paper Outline

## Title
Comparative Evaluation and Feature Importance Analysis of Machine Learning Models for Phishing Website Detection

## Authors
Murshud Azimov, Javid Shukurov
Western Caspian University

---

## Abstract (150-200 words)
- Problem: Phishing websites are a cybersecurity threat
- Gap: Need for reproducible model comparison + interpretability
- Method: 6 ML models, multiple metrics, permutation feature importance
- Results: Random Forest achieved best F1-score of 0.9733
- Conclusion: ML effective for phishing detection; feature importance improves interpretability

## Keywords
Phishing detection, machine learning, cybersecurity, feature importance, classification, explainable AI

---

## 1. Introduction
- P1: Problem (phishing attacks, consequences)
- P2: Why ML? (limitations of blacklists)
- P3: Gap (need for comparison + interpretability)
- P4: This study's approach
- Contributions list (4 items)

## 2. Related Work
- 2.1 Traditional phishing detection (blacklists, heuristics)
- 2.2 ML-based phishing detection (SVM, RF, DT, NN studies)
- 2.3 Interpretability in cybersecurity (XAI, feature importance)

## 3. Methodology
- 3.1 Dataset description (UCI, 11055 instances, 30 features)
- 3.2 Preprocessing (binary encoding, scaling)
- 3.3 Train-test split (80/20, stratified)
- 3.4 Models (6 models with brief descriptions)
- 3.5 Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- 3.6 Feature importance method (permutation importance)

## 4. Experimental Setup
- Python, scikit-learn, versions
- Hardware (if relevant)
- Hyperparameters table
- Cross-validation setup (5-fold stratified)

## 5. Results
- Table I: Test set results
- Table II: Cross-validation results
- Figure 1: Confusion matrix
- Figure 2: Feature importance bar chart
- Table III: Top-k feature subset results

## 6. Discussion
- Best model analysis
- Why recall/F1 matter more than accuracy
- Feature importance interpretation
- Comparison with related work (if applicable)

## 7. Limitations
- Single dataset
- Pre-engineered features
- Offline evaluation
- No deep learning comparison
- No real-time deployment

## 8. Conclusion
- Summary of findings
- Main takeaway
- Future work directions

## References
- 10-15 references from Google Scholar
- UCI dataset citation
- scikit-learn citation
- Key phishing detection papers

---

## Writing Schedule (suggested)
- Days 1-2: Run experiments, collect results
- Days 3-4: Write Methodology + Experimental Setup
- Days 5-6: Write Results + create tables/figures
- Days 7-8: Write Introduction + Related Work
- Days 9-10: Write Discussion + Limitations + Conclusion
- Days 11-12: Write Abstract, polish all sections
- Days 13-14: Review, format, finalize
- Day 15: Final check, submit/upload
