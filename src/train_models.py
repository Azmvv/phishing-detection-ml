"""
A Comparative and Explainable Machine Learning Study for Phishing Website Detection

This script:
1. Loads the UCI Phishing Websites Dataset
2. Trains 6 supervised ML models
3. Evaluates them with multiple metrics
4. Performs cross-validation
5. Generates confusion matrix
6. Computes permutation feature importance
7. Runs feature subset (top-k) analysis
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance
from sklearn.base import clone

warnings.filterwarnings("ignore")


def load_dataset():
    """Load and prepare the UCI Phishing Websites Dataset."""
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)

    phishing = fetch_ucirepo(id=327)

    X = phishing.data.features.copy()
    y_raw = phishing.data.targets.iloc[:, 0].copy()

    # Phishing = 1, Legitimate = 0
    # Original encoding: -1 = phishing, 1 = legitimate
    y = (y_raw == -1).astype(int)

    print(f"Dataset shape: {X.shape}")
    print(f"\nClass distribution:")
    print(f"  Phishing (1): {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Legitimate (0): {(1-y).sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"\nFeatures: {list(X.columns)}")

    return X, y


def split_data(X, y):
    """Split data into train and test sets."""
    print("\n" + "=" * 60)
    print("STEP 2: Train-Test Split")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def define_models():
    """Define the ML models to evaluate."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),

        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf", probability=True,
                class_weight="balanced", random_state=42
            ))
        ]),

        "Decision Tree": Pipeline([
            ("model", DecisionTreeClassifier(
                class_weight="balanced", random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("model", GradientBoostingClassifier(random_state=42))
        ])
    }

    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """Train all models and evaluate on test set."""
    print("\n" + "=" * 60)
    print("STEP 3: Training and Evaluating Models")
    print("=" * 60)

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")

        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1,
            "ROC-AUC": auc
        }

        print(f"    Accuracy: {acc:.4f} | Precision: {prec:.4f} | "
              f"Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    results_df = pd.DataFrame(results).T.sort_values(
        by="F1-score", ascending=False
    )

    return results_df, trained_models


def cross_validation(models, X_train, y_train):
    """Perform 5-fold stratified cross-validation."""
    print("\n" + "=" * 60)
    print("STEP 4: 5-Fold Stratified Cross-Validation")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results = {}

    for name, model in models.items():
        print(f"\n  Cross-validating {name}...")

        scores = cross_validate(
            model, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1
        )

        cv_results[name] = {
            "CV Accuracy": scores["test_accuracy"].mean(),
            "CV Precision": scores["test_precision"].mean(),
            "CV Recall": scores["test_recall"].mean(),
            "CV F1-score": scores["test_f1"].mean(),
            "CV ROC-AUC": scores["test_roc_auc"].mean()
        }

        print(f"    CV F1: {cv_results[name]['CV F1-score']:.4f} | "
              f"CV AUC: {cv_results[name]['CV ROC-AUC']:.4f}")

    cv_results_df = pd.DataFrame(cv_results).T.sort_values(
        by="CV F1-score", ascending=False
    )

    return cv_results_df


def plot_confusion_matrix(best_model, best_model_name, X_test, y_test, output_dir):
    """Generate and save confusion matrix for the best model."""
    print("\n" + "=" * 60)
    print("STEP 5: Confusion Matrix")
    print("=" * 60)

    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    print(f"  Confusion matrix saved to {output_dir}/confusion_matrix.png")
    print(f"\n  Classification Report ({best_model_name}):")
    print(classification_report(
        y_test, y_pred_best,
        target_names=["Legitimate", "Phishing"]
    ))


def compute_feature_importance(best_model, best_model_name, X_test, y_test, X_columns, output_dir):
    """Compute permutation feature importance."""
    print("\n" + "=" * 60)
    print("STEP 6: Permutation Feature Importance")
    print("=" * 60)

    perm = permutation_importance(
        best_model, X_test, y_test,
        scoring="f1",
        n_repeats=20,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": X_columns,
        "Importance": perm.importances_mean,
        "Std": perm.importances_std
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv(
        os.path.join(output_dir, "feature_importance.csv"), index=False
    )

    # Plot top 10 features
    top_n = 10
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=top_features,
        x="Importance", y="Feature",
        palette="viridis"
    )
    plt.title(f"Top {top_n} Most Important Features - {best_model_name}")
    plt.xlabel("Permutation Importance (F1-score)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300)
    plt.close()

    print(f"\n  Top {top_n} features:")
    for i, row in top_features.iterrows():
        print(f"    {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")

    print(f"\n  Feature importance saved to {output_dir}/")

    return importance_df


def feature_subset_analysis(best_model, best_model_name, importance_df,
                            X_train, X_test, y_train, y_test, output_dir):
    """Evaluate model performance with top-k features."""
    print("\n" + "=" * 60)
    print("STEP 7: Feature Subset Analysis (Top-k Features)")
    print("=" * 60)

    top_k_results = {}
    total_features = X_train.shape[1]

    for k in [5, 10, 15, total_features]:
        selected_features = importance_df.head(k)["Feature"].tolist()

        model_k = clone(best_model)
        model_k.fit(X_train[selected_features], y_train)

        y_pred_k = model_k.predict(X_test[selected_features])

        if hasattr(model_k, "predict_proba"):
            y_score_k = model_k.predict_proba(X_test[selected_features])[:, 1]
        else:
            y_score_k = model_k.decision_function(X_test[selected_features])

        label = f"Top {k}" if k < total_features else f"All {k}"

        top_k_results[label] = {
            "Accuracy": accuracy_score(y_test, y_pred_k),
            "Precision": precision_score(y_test, y_pred_k),
            "Recall": recall_score(y_test, y_pred_k),
            "F1-score": f1_score(y_test, y_pred_k),
            "ROC-AUC": roc_auc_score(y_test, y_score_k)
        }

        print(f"  {label} features -> F1: {top_k_results[label]['F1-score']:.4f} | "
              f"AUC: {top_k_results[label]['ROC-AUC']:.4f}")

    top_k_df = pd.DataFrame(top_k_results).T
    top_k_df.to_csv(os.path.join(output_dir, "top_k_feature_results.csv"))

    return top_k_df


def main():
    """Run the full experiment pipeline."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load data
    X, y = load_dataset()

    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Define and train models
    models = define_models()
    results_df, trained_models = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    # Save test results
    results_df.to_csv(os.path.join(output_dir, "test_results.csv"))

    # Step 4: Cross-validation
    cv_results_df = cross_validation(models, X_train, y_train)
    cv_results_df.to_csv(os.path.join(output_dir, "cv_results.csv"))

    # Identify best model
    best_model_name = results_df.index[0]
    best_model = trained_models[best_model_name]
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL (by F1-score): {best_model_name}")
    print(f"{'=' * 60}")

    # Step 5: Confusion matrix
    plot_confusion_matrix(best_model, best_model_name, X_test, y_test, output_dir)

    # Step 6: Feature importance
    importance_df = compute_feature_importance(
        best_model, best_model_name, X_test, y_test, X.columns, output_dir
    )

    # Step 7: Feature subset analysis
    top_k_df = feature_subset_analysis(
        best_model, best_model_name, importance_df,
        X_train, X_test, y_train, y_test, output_dir
    )

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nAll results saved to: {output_dir}/")
    print(f"  - test_results.csv")
    print(f"  - cv_results.csv")
    print(f"  - feature_importance.csv")
    print(f"  - top_k_feature_results.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - feature_importance.png")

    print(f"\n--- TEST SET RESULTS ---")
    print(results_df.to_string())

    print(f"\n--- CROSS-VALIDATION RESULTS ---")
    print(cv_results_df.to_string())

    print(f"\n--- TOP-K FEATURE RESULTS ({best_model_name}) ---")
    print(top_k_df.to_string())


if __name__ == "__main__":
    main()
