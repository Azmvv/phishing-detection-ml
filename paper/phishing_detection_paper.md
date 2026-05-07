# Comparative Evaluation and Feature Importance Analysis of Machine Learning Models for Phishing Website Detection

**Authors:** Murshud Azimov, Javid Shukurov  
**Affiliation:** Western Caspian University

---

## Abstract

Phishing websites are a major cybersecurity threat that deceive users into revealing sensitive information such as passwords, financial credentials, and personal data. Machine learning techniques have been widely explored for phishing website detection; however, comparing multiple models under the same experimental setting and analyzing their interpretability remains important. This study presents a comparative evaluation of six supervised machine learning models for phishing website detection using a public dataset from the UCI Machine Learning Repository. We evaluate Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, and Gradient Boosting classifiers. The models are assessed using accuracy, precision, recall, F1-score, and ROC-AUC. In addition, permutation feature importance is used to identify the most influential website features in the classification process. The results show that Random Forest achieved the best overall performance with an F1-score of 0.9733 and ROC-AUC of 0.9967. A feature subset analysis further demonstrates that using only the top 15 features yields performance comparable to using all 30 features. The findings suggest that machine learning models can provide effective and interpretable support for phishing detection.

**Keywords:** Phishing detection, machine learning, cybersecurity, feature importance, classification, explainable AI

---

## 1. Introduction

Phishing is one of the most prevalent cybersecurity threats, in which attackers create deceptive websites designed to steal sensitive user information such as login credentials, credit card numbers, and personal data. According to recent reports, phishing attacks continue to increase in both frequency and sophistication, causing significant financial losses and privacy violations for individuals and organizations worldwide.

Traditional defense mechanisms against phishing websites primarily rely on blacklist-based approaches, where known malicious URLs are maintained in a database and blocked when users attempt to access them. However, these methods have a fundamental limitation: they cannot detect newly created phishing websites that have not yet been reported and added to the blacklist. Since attackers frequently change domains and URL structures, blacklist-based systems often fail to provide timely protection against zero-day phishing attacks.

Machine learning-based approaches have emerged as a promising alternative for phishing website detection [3], [4]. By learning patterns from URL structure, domain characteristics, and webpage behavior features, machine learning models can potentially identify phishing websites even when they have not been previously encountered. Various supervised learning algorithms have been applied to this problem, including Decision Trees, Support Vector Machines, Random Forests, and ensemble methods [5], [10], [11]. However, a systematic comparison of multiple models under a reproducible experimental setup remains valuable, particularly when considering the trade-offs between different evaluation metrics and the interpretability of model decisions [4], [6].

In cybersecurity applications, understanding why a model classifies a website as phishing is important for security analysts who need to validate and act upon automated decisions [8]. Therefore, combining predictive performance with interpretability analysis provides a more complete picture of model utility in real-world phishing detection scenarios.

This study compares six supervised machine learning models for phishing website detection using a public dataset. In addition to performance evaluation using multiple metrics, we apply permutation feature importance to analyze which website features contribute most to phishing classification decisions. We also investigate whether a reduced feature subset can achieve comparable performance to using all available features.

The main contributions of this study are:

1. A comparative evaluation of six supervised machine learning models for phishing website detection under a consistent experimental setup.
2. A performance analysis using accuracy, precision, recall, F1-score, and ROC-AUC, with emphasis on metrics relevant to security-sensitive classification.
3. An interpretability analysis based on permutation feature importance to identify the most influential website features.
4. A feature subset analysis demonstrating that comparable performance can be achieved with fewer features.
5. A reproducible implementation using Python and scikit-learn, publicly available on GitHub.

---

## 2. Related Work

### 2.1 Traditional Phishing Detection

Early approaches to phishing detection relied primarily on blacklists and heuristic rules. Blacklist-based systems maintain databases of known phishing URLs and block access when a match is found. While effective for known threats, these systems cannot detect newly created phishing websites. Khonji et al. [1] provided a comprehensive literature survey on phishing detection methods, categorizing approaches into list-based, heuristic-based, visual similarity-based, and machine learning-based techniques. They highlighted that traditional methods struggle with zero-day phishing attacks due to their reliance on previously known patterns. Basit et al. [2] further discussed the taxonomy of phishing detection approaches, noting that heuristic-based methods require manual rule engineering and may not generalize well to evolving attack strategies.

### 2.2 Machine Learning-Based Phishing Detection

Machine learning approaches have been extensively studied for phishing website detection. Sahingoz et al. [3] applied multiple machine learning algorithms including Random Forest, Decision Tree, and SVM to phishing URL detection, demonstrating that ensemble methods generally outperform individual classifiers. Their work emphasized the importance of feature engineering from URL characteristics.

Chiew et al. [4] presented a comprehensive survey of machine learning-based solutions for phishing website detection, reviewing various feature types (URL-based, content-based, and third-party service-based) and classification algorithms used in the literature. They noted that while many studies report high accuracy, direct comparison across studies is difficult due to differences in datasets, feature sets, and evaluation protocols.

Subasi and Kremic [5] compared multiple machine learning classifiers for phishing website detection using the UCI dataset, reporting that Random Forest and Gradient Boosting achieved the highest classification performance. Their findings align with the general observation that ensemble methods are well-suited for this task.

Do et al. [6] investigated deep learning approaches for phishing detection, providing a taxonomy of current challenges and future directions. They noted that while deep learning can achieve high accuracy, traditional machine learning methods remain competitive when well-engineered features are available.

Alsariera et al. [7] explored phishing detection using machine learning techniques with a focus on feature selection and model comparison, demonstrating that careful feature engineering can significantly impact detection performance.

### 2.3 Interpretability in Cybersecurity

Recent work in machine learning has emphasized the importance of model interpretability, particularly in high-stakes domains such as cybersecurity. Arrieta et al. [8] provided a comprehensive survey on explainable artificial intelligence (XAI), discussing various techniques including feature importance, SHAP values, and LIME that can be applied to improve model transparency.

In the context of phishing detection, interpretability is important because security analysts need to understand why a model classifies a website as phishing in order to validate decisions and take appropriate action. Explainable approaches help bridge the gap between automated detection and human decision-making in cybersecurity operations.

### 2.4 Evaluation Methodology

Kohavi [9] established best practices for model evaluation through cross-validation, demonstrating that stratified 10-fold cross-validation provides reliable estimates of model performance. Following these established practices ensures that reported results are robust and reproducible.

---

## 3. Methodology

### 3.1 Dataset

This study uses the UCI Phishing Websites Dataset [12], which contains 11,055 website instances described by 30 input features. The features represent various characteristics of URLs, domains, and webpage behavior that are relevant to distinguishing phishing websites from legitimate ones. The target variable indicates whether a website is phishing or legitimate.

The dataset features include URL-based attributes (e.g., having_IP_Address, URL_Length, Shortining_Service, having_At_Symbol, prefix_suffix), domain-based attributes (e.g., Domain_registration_length, age_of_domain, DNSRecord), and page-based attributes (e.g., SSLfinal_State, Request_URL, URL_of_Anchor, Links_in_tags, SFH, web_traffic).

The original target encoding uses -1 for phishing and 1 for legitimate websites. For this study, the target was transformed into a binary format where phishing websites are encoded as 1 (positive class) and legitimate websites as 0. This encoding ensures that precision, recall, and F1-score are computed with respect to the phishing class, which is the class of primary interest in a detection system.

### 3.2 Preprocessing

The dataset did not require extensive preprocessing because all features were already encoded as integer values. No missing values were present. For models sensitive to feature scaling (Logistic Regression, K-Nearest Neighbors, and Support Vector Machine), standardization was applied using StandardScaler to transform features to zero mean and unit variance. Tree-based models (Decision Tree, Random Forest, and Gradient Boosting) were trained without scaling, as they are invariant to monotonic transformations of input features.

### 3.3 Train-Test Split

The dataset was divided into 80% training (8,844 instances) and 20% testing (2,211 instances) subsets using stratified sampling to preserve the class distribution in both subsets. A fixed random seed (random_state=42) was used to ensure reproducibility.

### 3.4 Models

Six supervised learning algorithms were evaluated:

- **Logistic Regression (LR):** A linear model with balanced class weights and a maximum of 1000 iterations.
- **K-Nearest Neighbors (KNN):** A distance-based classifier with k=5 neighbors.
- **Support Vector Machine (SVM-RBF):** A kernel-based classifier using the radial basis function kernel with balanced class weights.
- **Decision Tree (DT):** A tree-based classifier with balanced class weights.
- **Random Forest (RF):** An ensemble of 300 decision trees with balanced class weights.
- **Gradient Boosting (GB):** A sequential ensemble method using boosted decision trees.

All models were implemented using scikit-learn [13] with default hyperparameters except where noted above. Pipelines were used to combine preprocessing (scaling) and classification steps where appropriate.

### 3.5 Evaluation Metrics

The models were evaluated using five metrics:

- **Accuracy:** The proportion of correctly classified instances.
- **Precision:** The proportion of predicted phishing websites that are actually phishing (minimizes false positives).
- **Recall:** The proportion of actual phishing websites that are correctly detected (minimizes false negatives).
- **F1-score:** The harmonic mean of precision and recall, providing a balanced measure.
- **ROC-AUC:** The area under the Receiver Operating Characteristic curve, measuring the model's ability to discriminate between classes across all thresholds.

In phishing detection, recall is particularly important because a false negative (a phishing website classified as legitimate) may expose users to security risks. Therefore, the F1-score was used as the primary metric for model selection, as it balances precision and recall.

### 3.6 Cross-Validation

In addition to the held-out test set evaluation, 5-fold stratified cross-validation was performed on the training set to obtain more robust performance estimates and assess model stability.

### 3.7 Feature Importance Analysis

Permutation feature importance was applied to the best-performing model on the test set using F1-score as the scoring function. This method measures the decrease in model performance when each feature's values are randomly shuffled, thereby breaking the relationship between the feature and the target. Features that cause a larger performance drop when permuted are considered more important. Twenty repetitions were used to obtain stable importance estimates.

### 3.8 Feature Subset Analysis

To investigate whether a smaller feature subset can achieve comparable performance, the best model was retrained using only the top-k most important features (k = 5, 10, 15) and compared against the full feature set (k = 30).

---

## 4. Results

### 4.1 Test Set Performance

Table I presents the performance of all six models on the held-out test set, sorted by F1-score.

**Table I. Test Set Performance of Machine Learning Models**

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.9765 | 0.9803 | 0.9663 | 0.9733 | 0.9967 |
| Decision Tree | 0.9692 | 0.9740 | 0.9561 | 0.9650 | 0.9762 |
| SVM-RBF | 0.9566 | 0.9576 | 0.9439 | 0.9507 | 0.9901 |
| Gradient Boosting | 0.9502 | 0.9494 | 0.9378 | 0.9435 | 0.9913 |
| KNN | 0.9471 | 0.9499 | 0.9296 | 0.9397 | 0.9856 |
| Logistic Regression | 0.9317 | 0.9295 | 0.9153 | 0.9224 | 0.9786 |

Random Forest achieved the highest performance across all metrics, with an F1-score of 0.9733 and ROC-AUC of 0.9967. Decision Tree ranked second with an F1-score of 0.9650. All models achieved accuracy above 93%, indicating that the phishing detection task is well-suited for supervised learning approaches.

### 4.2 Cross-Validation Results

Table II presents the 5-fold stratified cross-validation results on the training set.

**Table II. 5-Fold Stratified Cross-Validation Results**

| Model | CV Accuracy | CV Precision | CV Recall | CV F1-score | CV ROC-AUC |
|-------|-------------|--------------|-----------|-------------|------------|
| Random Forest | 0.9692 | 0.9675 | 0.9630 | 0.9652 | 0.9946 |
| Decision Tree | 0.9562 | 0.9499 | 0.9515 | 0.9507 | 0.9680 |
| Gradient Boosting | 0.9487 | 0.9502 | 0.9331 | 0.9415 | 0.9899 |
| SVM-RBF | 0.9481 | 0.9490 | 0.9331 | 0.9409 | 0.9868 |
| KNN | 0.9375 | 0.9345 | 0.9237 | 0.9290 | 0.9804 |
| Logistic Regression | 0.9254 | 0.9163 | 0.9153 | 0.9157 | 0.9788 |

The cross-validation results are consistent with the test set results, confirming that Random Forest is the best-performing model. The consistency between test set and cross-validation performance suggests that the results are not due to a favorable train-test split.

### 4.3 Confusion Matrix Analysis

Figure 1 shows the confusion matrix for the Random Forest model on the test set.

**[Figure 1: Confusion Matrix - Random Forest]**

The confusion matrix reveals:
- **True Negatives (Legitimate correctly classified):** 1,212
- **False Positives (Legitimate misclassified as Phishing):** 19
- **False Negatives (Phishing misclassified as Legitimate):** 33
- **True Positives (Phishing correctly classified):** 947

Out of 980 actual phishing websites in the test set, the Random Forest model correctly identified 947 (96.6% recall). Only 33 phishing websites were misclassified as legitimate, representing potential security risks. On the legitimate side, only 19 out of 1,231 legitimate websites were incorrectly flagged as phishing.

### 4.4 Feature Importance Analysis

Table III presents the top 10 most important features identified by permutation feature importance.

**Table III. Top 10 Most Important Features (Random Forest)**

| Rank | Feature | Importance | Std |
|------|---------|------------|-----|
| 1 | sslfinal_state | 0.1232 | 0.0064 |
| 2 | url_of_anchor | 0.1172 | 0.0045 |
| 3 | prefix_suffix | 0.0396 | 0.0042 |
| 4 | web_traffic | 0.0311 | 0.0035 |
| 5 | links_in_tags | 0.0292 | 0.0029 |
| 6 | having_sub_domain | 0.0242 | 0.0026 |
| 7 | age_of_domain | 0.0144 | 0.0023 |
| 8 | links_pointing_to_page | 0.0127 | 0.0027 |
| 9 | sfh | 0.0121 | 0.0023 |
| 10 | dnsrecord | 0.0103 | 0.0015 |

The two most influential features are `sslfinal_state` (SSL certificate status) and `url_of_anchor` (proportion of anchor URLs pointing to different domains), with importance values substantially higher than the remaining features. This suggests that SSL-related characteristics and external link patterns are the strongest indicators for distinguishing phishing websites from legitimate ones.

### 4.5 Feature Subset Analysis

Table IV shows the performance of the Random Forest model when trained with different numbers of top features.

**Table IV. Feature Subset Analysis (Random Forest)**

| Feature Set | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------------|----------|-----------|--------|----------|---------|
| Top 5 | 0.9263 | 0.9147 | 0.9194 | 0.9170 | 0.9811 |
| Top 10 | 0.9502 | 0.9430 | 0.9449 | 0.9439 | 0.9900 |
| Top 15 | 0.9697 | 0.9672 | 0.9643 | 0.9658 | 0.9944 |
| All 30 | 0.9747 | 0.9773 | 0.9653 | 0.9713 | 0.9965 |

Using only the top 15 features achieves an F1-score of 0.9658, which is only 0.0055 lower than using all 30 features (0.9713). This indicates that a reduced feature set can provide nearly equivalent performance while potentially simplifying the detection system. However, using only the top 5 features results in a more noticeable performance drop (F1-score of 0.9170), suggesting that at least 10-15 features are needed for robust classification.

Note: The slight difference between the "All 30" row in Table IV (F1: 0.9713) and the Random Forest result in Table I (F1: 0.9733) is due to the model being retrained from scratch for the feature subset experiment, which may produce minor variations despite using the same random seed.

---

## 5. Discussion

### 5.1 Model Comparison

The experimental results demonstrate that ensemble-based models, particularly Random Forest, outperform simpler baseline models for phishing website detection. Random Forest achieved the highest F1-score (0.9733) and ROC-AUC (0.9967), indicating both strong classification performance and excellent discrimination ability across all decision thresholds.

Interestingly, Decision Tree performed as the second-best model (F1-score: 0.9650), outperforming both SVM-RBF and Gradient Boosting. This may be because the dataset features are already well-engineered categorical/ordinal variables that align well with tree-based decision boundaries. The strong performance of tree-based models suggests that the phishing classification problem involves clear decision rules based on feature thresholds.

Logistic Regression, while achieving the lowest F1-score (0.9224), still provides reasonable performance and serves as a useful interpretable baseline. Its lower performance likely reflects the presence of nonlinear relationships between features that linear models cannot capture.

### 5.2 Importance of Metric Selection

In phishing detection, accuracy alone is insufficient for model evaluation. A model with high accuracy but low recall may miss a significant number of phishing websites, exposing users to security risks. The Random Forest model achieved a recall of 0.9663, meaning it correctly identified approximately 96.6% of phishing websites. The 33 false negatives represent cases where phishing websites were incorrectly classified as legitimate—these are the most dangerous errors in a phishing detection system.

The choice of F1-score as the primary selection metric balances the need to detect phishing websites (recall) with the need to avoid incorrectly blocking legitimate websites (precision). In deployment scenarios where user trust is critical, a high false positive rate could lead users to ignore warnings, while a high false negative rate directly exposes users to attacks.

### 5.3 Feature Importance Interpretation

The feature importance analysis reveals that SSL certificate status (`sslfinal_state`) and anchor URL patterns (`url_of_anchor`) are the two most influential features for phishing classification. This finding is consistent with known phishing characteristics:

- **SSL certificate status:** Phishing websites often lack valid SSL certificates or use self-signed certificates, as obtaining legitimate certificates requires domain verification that may expose the attacker's identity.
- **URL of anchor:** Phishing websites frequently contain links pointing to external domains or use empty/suspicious anchor references, as the page content is often copied from legitimate sites but hosted on different infrastructure.

Other important features include `prefix_suffix` (use of hyphens in domain names, common in phishing URLs), `web_traffic` (phishing sites typically have low traffic rankings), and `having_sub_domain` (excessive subdomain usage is a common phishing technique).

These findings provide actionable insights for security analysts and can inform the design of rule-based detection systems that complement machine learning approaches.

### 5.4 Feature Subset Efficiency

The feature subset analysis demonstrates that the Random Forest model can achieve 99.4% of its full performance (F1-score 0.9658 vs. 0.9713) using only the top 15 features. This has practical implications for real-time phishing detection systems where feature extraction time and computational resources may be constrained. A system using fewer features could provide faster classification while maintaining acceptable accuracy.

---

## 6. Limitations

This study has several limitations that should be considered when interpreting the results:

1. **Single dataset:** The experiments were conducted on a single public dataset (UCI Phishing Websites Dataset), which may not fully represent the diversity of current phishing techniques. Phishing attacks evolve rapidly, and features that are discriminative today may become less effective as attackers adapt.

2. **Pre-engineered features:** The dataset contains pre-engineered integer-encoded features rather than raw URLs or webpage content. This limits the ability to explore feature engineering approaches and may not reflect the full complexity of real-world phishing detection.

3. **Offline evaluation:** The models were evaluated in an offline setting using a static dataset. Real-world phishing detection systems must handle concept drift, where the characteristics of phishing websites change over time.

4. **Default hyperparameters:** Most models were trained with default or minimally tuned hyperparameters. Hyperparameter optimization could potentially improve the performance of underperforming models.

5. **No deep learning comparison:** This study focuses on traditional machine learning models and does not include deep learning approaches (e.g., neural networks operating on raw URLs or webpage content), which have shown promise in recent literature.

6. **No real-time deployment:** The models were not tested in a real-time detection scenario, where factors such as inference speed, feature extraction latency, and system integration become important.

Future work could address these limitations by evaluating models on more recent and diverse datasets, incorporating deep learning methods, performing hyperparameter optimization, and developing a real-time detection prototype.

---

## 7. Conclusion

This study presented a comparative evaluation of six supervised machine learning models for phishing website detection using the UCI Phishing Websites Dataset. The models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC, with emphasis on metrics relevant to security-sensitive classification tasks.

The results demonstrated that Random Forest achieved the best overall performance with an F1-score of 0.9733 and ROC-AUC of 0.9967, correctly identifying 96.6% of phishing websites in the test set. The consistency between test set and cross-validation results confirms the robustness of this finding.

Permutation feature importance analysis identified SSL certificate status and anchor URL patterns as the two most influential features for phishing classification, providing interpretable insights into model decision-making. The feature subset analysis further showed that using only the top 15 features achieves performance comparable to the full feature set, suggesting opportunities for computational efficiency in deployment.

The findings confirm that machine learning, particularly ensemble methods, can provide effective and interpretable support for phishing website detection. The reproducible implementation and publicly available code facilitate further research and extension of this work.

Future work may include testing the models on newer phishing datasets that reflect current attack techniques, applying deep learning methods for end-to-end learning from raw URLs, performing comprehensive hyperparameter optimization, and developing a real-time detection prototype.

---

## References

[1] M. Khonji, Y. Iraqi, and A. Jones, "Phishing Detection: A Literature Survey," *IEEE Communications Surveys & Tutorials*, vol. 15, no. 4, pp. 2091–2121, 2013. Available: https://www.sciencedirect.com/science/article/abs/pii/S1574013715000039

[2] A. Basit, M. Zafar, X. Liu, A. R. Javed, Z. Jalil, and K. Kifayat, "A Comprehensive Survey of AI-Enabled Phishing Attacks Detection Techniques," *Telecommunication Systems*, vol. 76, pp. 139–154, 2021. Available: https://link.springer.com/chapter/10.1007/978-1-4419-9326-7_5

[3] O. K. Sahingoz, E. Buber, O. Demir, and B. Diri, "Machine Learning Based Phishing Detection from URLs," *Expert Systems with Applications*, vol. 117, pp. 345–357, 2019.

[4] K. L. Chiew, K. S. C. Yong, and C. L. Tan, "A Survey of Machine Learning-Based Solutions for Phishing Website Detection," *Machine Learning and Knowledge Extraction*, vol. 3, no. 3, pp. 672–694, 2021. Available: https://www.mdpi.com/2504-4990/3/3/34

[5] A. Subasi and E. Kremic, "Comparison of Adaboost with MultiBoosting for Phishing Website Detection," *Procedia Computer Science*, vol. 168, pp. 272–278, 2020.

[6] N. Q. Do, A. Selamat, O. Krejcar, E. Herrera-Viedma, and H. Fujita, "Deep Learning for Phishing Detection: Taxonomy, Current Challenges and Future Directions," *IEEE Access*, vol. 10, pp. 36429–36463, 2022. Available: https://ieeexplore.ieee.org/abstract/document/9716113

[7] Y. A. Alsariera, V. E. Adeyemo, A. O. Balogun, and A. K. Alazzawi, "AI Meta-Learners and Extra-Trees Algorithm for the Detection of Phishing Websites," *IEEE Access*, vol. 10, pp. 42532–42542, 2022. Available: https://ieeexplore.ieee.org/abstract/document/9852458

[8] A. B. Arrieta et al., "Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI," *Information Fusion*, vol. 58, pp. 82–115, 2020. Available: https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/widm.1306

[9] R. Kohavi, "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection," in *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)*, vol. 2, pp. 1137–1143, 1995. Available: https://ai.stanford.edu/~ronnyk/editorial.pdf

[10] S. Chatterjee and A. S. Thakur, "Phishing Detection Using Machine Learning Techniques," arXiv preprint arXiv:2009.11116, 2020. Available: https://arxiv.org/abs/2009.11116

[11] A. K. Jain and B. B. Gupta, "A Machine Learning Based Approach for Phishing Detection Using Hyperlinks Information," *Journal of Ambient Intelligence and Humanized Computing*, vol. 10, pp. 2015–2028, 2019. Available: https://link.springer.com/article/10.1007/s13042-018-00906-1

[12] UCI Machine Learning Repository, "Phishing Websites Dataset," 2016. Available: https://archive.ics.uci.edu/dataset/327/phishing+websites

[13] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

## Appendix: Reproducibility

The complete source code for this study is available at: https://github.com/Azmvv/phishing-detection-ml

**Environment:**
- Python 3.x
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- ucimlrepo

All experiments use fixed random seeds (random_state=42) for reproducibility.
