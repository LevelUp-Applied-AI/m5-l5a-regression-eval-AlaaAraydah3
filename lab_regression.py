"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score)


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset.

    Returns:
        DataFrame with all columns.
    """
    # TODO: Load the CSV and return the DataFrame
    df = pd.read_csv(filepath)
    print(f"Data shape: {df.shape}")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDistribution of target column 'churned':")
    print(df['churned'].value_counts())
    return df
   


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification.

    Args:
        df: DataFrame with features and target.
        target_col: Name of the target column.
        test_size: Fraction for test set.
        random_state: Random seed.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # Separate features and target, then choose stratification safely
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = None
    if y.nunique() <= 10 and y.value_counts().min() >= 2:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    print(f"\nTrain set size: {X_train.shape[0]} rows")
    print(f"Test set size: {X_test.shape[0]} rows")

    if stratify is not None:
        print("\nTrain churn distribution:")
        print(y_train.value_counts)
        print("\nTest churn distribution:")
        print(y_test.value_counts)
    else:
        print("\nStratification was not applied because the target variable is not categorical or has too few samples per group.")

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression.

    Returns:
        sklearn Pipeline object.
    """
    # Create and return a Pipeline with two steps

    model= LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)
    scaler =  StandardScaler()
    pip = Pipeline([
        ('scaler', scaler),
        ('classifier',model)
    ])
    
    return pip


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression.

    Returns:
        sklearn Pipeline object.
    """
    # TODO: Create and return a Pipeline for Ridge regression
    pip =  Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])
    return pip


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics.

    Args:
        pipeline: sklearn Pipeline with a classifier.
        X_train, X_test: Feature arrays.
        y_train, y_test: Label arrays.

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
    """
    # TODO: Fit the pipeline on training data, predict on test, compute metrics
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics.

    Args:
        pipeline: sklearn Pipeline with a regressor.
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.

    Returns:
        Dictionary with keys: 'mae', 'r2'.
    """
    # TODO: Fit the pipeline, predict, and compute MAE and R²
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'mae': mae, 'r2': r2}


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation on the pipeline.

    Args:
        pipeline: sklearn Pipeline.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of folds.

    Returns:
        Array of cross-validation scores.
    """
    # TODO: Run cross_val_score with StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
    return scores


if __name__ == "__main__":
    # Load the telecom churn dataset
    df = load_data()
    if df is not None:
        # Print basic information about loaded data
        print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

        # Define numeric features for classification model
        numeric_features = ["tenure", "monthly_charges", "total_charges",
                           "num_support_calls", "senior_citizen",
                           "has_partner", "has_dependents"]

        # ===== CLASSIFICATION TASK: Predict Churn =====
        # Prepare data for classification by selecting features and target
        df_cls = df[numeric_features + ["churned"]].dropna()
        # Split data into train and test sets with stratification
        split = split_data(df_cls, "churned")
        if split:
            # Unpack the split data
            X_train, X_test, y_train, y_test = split
            # Build logistic regression pipeline
            pipe = build_logistic_pipeline()
            if pipe:
                # Train and evaluate classifier on test set
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                # Display classification metrics
                print(f"Logistic Regression: {metrics}")

                # Perform cross-validation with stratified k-fold
                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    # Print mean and std of cross-validation scores
                    print(f"CV: {scores.mean():.3f} +/- {scores.std():.3f}")

        # ===== REGRESSION TASK: Predict Monthly Charges =====
        # Prepare data for regression by selecting features and continuous target
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()
        # Split data into train and test sets without stratification (continuous target)
        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            # Unpack the split data for regression
            X_tr, X_te, y_tr, y_te = split_reg
            # Build ridge regression pipeline
            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                # Train and evaluate regressor on test set
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                # Display regression metrics
                print(f"Ridge Regression: {reg_metrics}")


"""
===============================================================================
TASK 7: SUMMARY OF FINDINGS
===============================================================================

1. IMPORTANT FEATURES FOR PREDICTING CHURN
   Based on the logistic regression model, the following features are likely
   most important for predicting customer churn:
   
   - tenure: Customers with longer tenure are less likely to churn. This reflects
     increased customer loyalty and switching costs.
   - monthly_charges: Higher monthly charges may correlate with churn if customers
     perceive poor value. This is directly related to price sensitivity.
   - total_charges: Cumulative spend reflects long-term customer value and lifetime
     engagement with the company.
   - num_support_calls: Frequent support calls could indicate either high engagement
     or unresolved issues leading to churn.
   - senior_citizen, has_partner, has_dependents: Demographic factors that influence
     stability and contract duration.

2. LOGISTIC REGRESSION MODEL PERFORMANCE
   The logistic regression pipeline (with StandardScaler and balanced class weights):
   
   - Handles feature scaling via StandardScaler to normalize numeric inputs.
   - Balanced class weights mitigate imbalanced churn distribution.
   - Stratified cross-validation ensures proper class representation in each fold.
   
   For CHURN PREDICTION, RECALL is typically more important than PRECISION:
   - Recall: Identifying truly churning customers (false negatives costly - lose revenue)
   - Precision: Minimizing false churn alerts (false positives - wasted retention effort)
   
   A missed churner (low recall) = lost revenue and customer.
   A false alarm (low precision) = extra retention offer cost.
   
   Recommendation: Prioritize high recall even if it means lower precision, as losing
   a customer is more costly than over-targeting retention campaigns.

3. NEXT STEPS TO IMPROVE PERFORMANCE
   
   a) Feature Engineering:
      - Create interaction features (e.g., monthly_charges * tenure)
      - Develop tenure-based segments (new, established, long-term customers)
      - Add calculated ratios (e.g., monthly_charges / total_charges)
   
   b) Model Improvements:
      - Experiment with other classifiers (Random Forest, Gradient Boosting, SVM)
      - Tune hyperparameters for LogisticRegression (C, solver, max_iter)
      - Use GridSearchCV for systematic hyperparameter optimization
   
 

===============================================================================
"""
