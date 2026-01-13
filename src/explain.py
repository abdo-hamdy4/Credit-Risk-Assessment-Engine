# src/explain.py
"""
Explainability Module for Credit Risk Assessment Engine.

=== WHY SHAP? ===

REGULATORY COMPLIANCE (2026 KEY):
- EU AI Act: High-risk AI systems (including credit scoring) require explainability
- US ECOA/FCRA: Adverse action notices must provide specific reasons for denial
- Basel III: Model risk management requires interpretable models

SHAP ADVANTAGES:
1. THEORETICAL FOUNDATION: Based on Shapley values from cooperative game theory
   - Each feature's contribution is fairly distributed
   - Satisfies efficiency, symmetry, dummy, and additivity axioms

2. MODEL-AGNOSTIC: Works with any ML model (we use TreeSHAP for XGBoost efficiency)

3. LOCAL + GLOBAL: Explains individual predictions AND overall feature importance

4. DIRECTIONAL: Shows whether a feature INCREASES or DECREASES risk

MATHEMATICAL INTUITION:
SHAP value for feature i = average marginal contribution of feature i
                         across all possible feature coalitions

For a prediction: Σ(SHAP values) + base_value = model_output
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


def load_model_and_metadata(model_dir: str) -> Tuple:
    """
    Load the trained model and associated metadata.
    
    Returns: (model, feature_names, metadata, encoders)
    """
    model_path = os.path.join(model_dir, 'xgb_credit_risk.pkl')
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    encoders_path = os.path.join(model_dir, 'encoders.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    
    metadata = None
    feature_names = None
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_names = metadata.get('feature_names', [])
    
    # Load encoders if available
    encoders = None
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)
    
    return model, feature_names, metadata, encoders


def create_shap_explainer(model):
    """
    Create SHAP TreeExplainer for XGBoost model.
    
    WHY TREEEXPLAINER?
    - Exact Shapley values in polynomial time (vs exponential for KernelSHAP)
    - Specifically optimized for tree-based models
    - Order of magnitude faster than model-agnostic approaches
    """
    explainer = shap.TreeExplainer(model)
    return explainer


def compute_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for given instances.
    
    Returns array of shape (n_samples, n_features)
    Each value represents the feature's contribution to moving the
    prediction away from the base value (average prediction).
    """
    shap_values = explainer.shap_values(X)
    return shap_values


def generate_reason_for_decision(
    model,
    explainer,
    applicant_data: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 5,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a "Reason for Decision" explanation for a single applicant.
    
    This function creates both:
    1. A visual SHAP force plot showing feature contributions
    2. A text-based explanation suitable for regulatory documentation
    
    Parameters
    ----------
    model : XGBClassifier
        Trained model
    explainer : shap.TreeExplainer
        SHAP explainer object
    applicant_data : pd.DataFrame
        Single-row DataFrame with applicant features
    feature_names : list
        Names of features used in model
    top_n : int
        Number of top contributing factors to highlight
    save_path : str, optional
        Path to save the visualization
        
    Returns
    -------
    str : Text explanation of the decision
    """
    if applicant_data.shape[0] != 1:
        raise ValueError("applicant_data must contain exactly one row")
    
    # Get prediction and probability
    prob_default = model.predict_proba(applicant_data)[0, 1]
    prediction = "HIGH RISK (Default Likely)" if prob_default > 0.5 else "LOW RISK (Default Unlikely)"
    
    # Compute SHAP values for this applicant
    shap_values = explainer.shap_values(applicant_data)
    base_value = explainer.expected_value
    
    # Handle case where shap_values might be a list (for multiclass)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take the positive class
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1]
    
    # Get SHAP values as flat array
    shap_vals = shap_values[0] if shap_values.ndim > 1 else shap_values
    
    # Create feature importance ranking
    feature_effects = list(zip(feature_names, shap_vals, applicant_data.values[0]))
    feature_effects_sorted = sorted(feature_effects, key=lambda x: abs(x[1]), reverse=True)
    
    # Build text explanation
    explanation_lines = [
        "=" * 60,
        "CREDIT RISK DECISION EXPLANATION",
        "=" * 60,
        f"\nOVERALL ASSESSMENT: {prediction}",
        f"Default Probability: {prob_default:.1%}",
        f"\nBase Risk Level: {base_value:.3f}",
        "\n" + "-" * 40,
        f"TOP {top_n} FACTORS INFLUENCING THIS DECISION:",
        "-" * 40
    ]
    
    for i, (feature, shap_val, actual_val) in enumerate(feature_effects_sorted[:top_n], 1):
        direction = "INCREASES risk ↑" if shap_val > 0 else "DECREASES risk ↓"
        explanation_lines.append(
            f"\n{i}. {feature.upper()}"
            f"\n   Value: {actual_val}"
            f"\n   Impact: {direction} (SHAP: {shap_val:+.4f})"
        )
    
    explanation_lines.extend([
        "\n" + "-" * 40,
        "INTERPRETATION GUIDE:",
        "-" * 40,
        "• Positive SHAP values push prediction toward DEFAULT",
        "• Negative SHAP values push prediction toward NO DEFAULT",
        "• Larger absolute values = stronger influence on decision",
        "\n" + "=" * 60
    ])
    
    explanation_text = "\n".join(explanation_lines)
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    shap.force_plot(
        base_value,
        shap_vals,
        applicant_data.iloc[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot - {prediction}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return explanation_text


def generate_global_explanation(
    model,
    X: pd.DataFrame,
    feature_names: List[str],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate global feature importance using SHAP.
    
    This shows which features are most important OVERALL across all predictions.
    Useful for model validation and regulatory documentation.
    """
    explainer = create_shap_explainer(model)
    shap_values = explainer.shap_values(X)
    
    # Handle multiclass output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Create summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title("Global Feature Importance (SHAP Summary)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Global explanation saved to: {save_path}")
    
    return importance_df


def explain_single_applicant(applicant_data: pd.DataFrame, model_dir: str = None) -> str:
    """
    Convenience function to explain a single applicant's risk assessment.
    
    Parameters
    ----------
    applicant_data : pd.DataFrame
        Single-row DataFrame with applicant features
    model_dir : str
        Directory containing the trained model
        
    Returns
    -------
    str : Text explanation
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    
    model, feature_names, metadata = load_model_and_metadata(model_dir)
    explainer = create_shap_explainer(model)
    
    explanation = generate_reason_for_decision(
        model, explainer, applicant_data, feature_names
    )
    
    return explanation


if __name__ == "__main__":
    # Example usage with dummy data
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
    
    if os.path.exists(os.path.join(model_dir, 'xgb_credit_risk.pkl')):
        model, feature_names, metadata = load_model_and_metadata(model_dir)
        print(f"Loaded model with {len(feature_names)} features")
        print(f"Features: {feature_names[:5]}...")
        
        # Create dummy applicant for demonstration
        dummy_applicant = pd.DataFrame({
            feat: [0.5] for feat in feature_names
        })
        
        explainer = create_shap_explainer(model)
        explanation = generate_reason_for_decision(
            model, explainer, dummy_applicant, feature_names
        )
        print(explanation)
    else:
        print(f"Model not found in {model_dir}")
        print("Please run train.py first to train the model.")
