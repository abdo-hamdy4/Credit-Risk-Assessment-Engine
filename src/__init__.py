# src/__init__.py
"""
Credit Risk Assessment Engine - Source Package

This package contains modular utilities for:
- Data preprocessing (preprocess.py)
- Model training with XGBoost (train.py)
- SHAP-based explainability (explain.py)
"""

# Export functions from explain module for easier imports
from src.explain import (
    load_model_and_metadata,
    create_shap_explainer,
    compute_shap_values,
    generate_reason_for_decision,
    generate_global_explanation,
    explain_single_applicant
)
