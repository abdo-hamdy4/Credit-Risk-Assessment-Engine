# src/train.py
"""
Model Training Module for Credit Risk Assessment Engine.

=== WHY XGBOOST? ===

1. PERFORMANCE: Consistently top performer on tabular financial data (Kaggle, industry benchmarks)
2. HANDLES MIXED DATA: Native support for numeric + encoded categorical features
3. REGULARIZATION: L1/L2 penalties prevent overfitting on imbalanced credit data
4. INTERPRETABILITY: Tree-based structure works seamlessly with SHAP
5. PRODUCTION-READY: Fast inference, handles missing values, widely deployed in fintech

=== WHY RANDOMIZEDSEARCHCV? ===

- Grid search is O(n^k) where k = number of hyperparameters
- Randomized search samples from distributions, often finds optimal region faster
- For 30 iterations with 5-fold CV, we get 150 model fits vs potentially thousands
- Research shows random search competitive with grid search in most cases (Bergstra & Bengio, 2012)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import uniform, randint

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# Import preprocessing utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import run_preprocessing_pipeline


def create_xgb_model(random_state: int = 42) -> XGBClassifier:
    """
    Create base XGBoost classifier with sensible defaults.
    
    PARAMETER CHOICES:
    - objective='binary:logistic': Standard for binary classification
    - eval_metric='logloss': Probabilistic metric, better for risk scoring
    - use_label_encoder=False: Avoid deprecation warnings
    - scale_pos_weight: Will be set dynamically based on class imbalance
    """
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0
    )
    return model


def get_hyperparameter_space() -> dict:
    """
    Define hyperparameter search space for XGBoost.
    
    PARAMETER RATIONALE:
    - n_estimators (100-500): More trees = better fit, but diminishing returns past ~300
    - max_depth (3-10): Deeper trees = more complex interactions, risk of overfitting
    - learning_rate (0.01-0.3): Lower = more robust but slower training
    - subsample (0.6-1.0): Row sampling, prevents overfitting
    - colsample_bytree (0.6-1.0): Feature sampling, reduces correlation between trees
    - min_child_weight (1-10): Regularization, higher = more conservative
    - gamma (0-5): Minimum loss reduction for split, regularization
    - reg_alpha (0-1): L1 regularization
    - reg_lambda (0-1): L2 regularization
    """
    param_space = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.29),  # 0.01 to 0.30
        'subsample': uniform(0.6, 0.4),        # 0.6 to 1.0
        'colsample_bytree': uniform(0.6, 0.4), # 0.6 to 1.0
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }
    return param_space


def train_with_hyperparameter_tuning(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    n_iter: int = 30,
    cv_folds: int = 5,
    random_state: int = 42
) -> tuple:
    """
    Train XGBoost with RandomizedSearchCV for hyperparameter tuning.
    
    WHY 30 ITERATIONS, 5 FOLDS?
    - 30 iterations: Good coverage of hyperparameter space without excessive compute
    - 5-fold CV: Standard in industry, balances variance estimation vs compute cost
    - Stratified folds: Maintains class distribution in each fold
    
    Returns: (best_model, best_params, cv_results)
    """
    print("=" * 60)
    print("HYPERPARAMETER TUNING WITH RANDOMIZEDSEARCHCV")
    print("=" * 60)
    print(f"Iterations: {n_iter} | CV Folds: {cv_folds}")
    print(f"Total model fits: {n_iter * cv_folds}")
    
    # Create base model
    base_model = create_xgb_model(random_state)
    
    # Get search space
    param_space = get_hyperparameter_space()
    
    # Stratified K-Fold to maintain class balance
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # RandomizedSearchCV
    # WHY SCORING='ROC_AUC'? 
    # - AUC is threshold-independent, better for probability calibration
    # - Critical for credit risk where we need well-calibrated probabilities
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    
    print("\nStarting hyperparameter search...")
    random_search.fit(X_train, y_train)
    
    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETERS FOUND:")
    print("=" * 60)
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nBest CV ROC-AUC: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.cv_results_


def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Comprehensive model evaluation on test set.
    
    METRICS EXPLAINED:
    - Accuracy: Overall correctness (can be misleading with imbalanced data)
    - Precision: Of predicted defaults, how many were actual defaults? (False positive cost)
    - Recall: Of actual defaults, how many did we catch? (False negative cost)
    - F1: Harmonic mean of precision and recall
    - ROC-AUC: Probability ranking quality, threshold-independent
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))
    
    metrics['confusion_matrix'] = cm.tolist()
    return metrics


def save_model(model: XGBClassifier, feature_names: list, metrics: dict, 
               best_params: dict, output_dir: str) -> str:
    """
    Save trained model and metadata for deployment.
    
    SAVES:
    - model.pkl: Serialized XGBoost model
    - model_metadata.json: Feature names, hyperparameters, metrics, timestamp
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'xgb_credit_risk.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'model_name': 'XGBoost Credit Risk Classifier',
        'training_date': datetime.now().isoformat(),
        'feature_names': feature_names,
        'best_hyperparameters': best_params,
        'test_metrics': {k: v for k, v in metrics.items() if k != 'confusion_matrix'},
        'confusion_matrix': metrics.get('confusion_matrix')
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata saved to: {metadata_path}")
    
    return model_path


def run_training_pipeline(data_path: str, output_dir: str = None) -> tuple:
    """
    Execute the full training pipeline.
    
    Returns: (trained_model, metrics, feature_names)
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    
    # Step 1: Preprocessing
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)
    X_train, X_test, y_train, y_test, encoders, feature_names = run_preprocessing_pipeline(data_path)
    
    # Step 2: Hyperparameter tuning
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)
    best_model, best_params, cv_results = train_with_hyperparameter_tuning(X_train, y_train)
    
    # Step 3: Evaluation
    metrics = evaluate_model(best_model, X_test, y_test)
    
    # Step 4: Save model
    save_model(best_model, feature_names, metrics, best_params, output_dir)
    
    # Save encoders for inference
    encoders_path = os.path.join(output_dir, 'encoders.pkl')
    joblib.dump(encoders, encoders_path)
    print(f"Encoders saved to: {encoders_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    
    return best_model, metrics, feature_names


if __name__ == "__main__":
    # Execute training pipeline
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'loan.csv'))
    
    if os.path.exists(data_path):
        model, metrics, features = run_training_pipeline(data_path)
        print(f"\nFinal ROC-AUC: {metrics['roc_auc']:.4f}")
    else:
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please place loan.csv in the data/ folder and run again.")
