# src/preprocess.py
"""
Data Preprocessing Module for Credit Risk Assessment Engine.

=== WHY THESE CHOICES? ===

MISSING VALUE HANDLING:
In financial datasets, missingness is rarely random. A missing income field might 
indicate self-employment or unreported sources. We use:
  - Median imputation for numerical fields (robust to outliers common in income data)
  - "Missing" category for categorical fields (preserves the signal that data was unavailable)

FEATURE ENGINEERING:
1. Debt-to-Income Ratio (DTI): Classic lending metric. Higher DTI = higher strain on repayment.
   Formula: total_debt / annual_income
   
2. Credit Utilization: Percentage of available credit being used. High utilization (>30%) 
   is a strong default predictor per FICO scoring methodology.
   Formula: revolving_balance / credit_limit
   
3. Stability Index: Combines employment tenure and home ownership status.
   Rationale: Stable borrowers (long employment + homeownership) historically default less.
   Formula: 0.6 * normalized_emp_length + 0.4 * home_ownership_score

CLASS IMBALANCE:
Credit default is rare (typically 5-15%). Training on raw data biases models toward 
predicting "no default". SMOTE synthesizes minority class examples to rebalance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load the LendingClub dataset from CSV with memory optimization.
    
    Handles large datasets by:
    1. Loading only the columns we need
    2. Reading in chunks to avoid memory overflow
    3. Using optimized data types
    
    Parameters
    ----------
    filepath : str
        Path to loan.csv file
    sample_size : int, optional
        If provided, limit the number of rows loaded. Useful for testing/debugging.
        
    Returns
    -------
    pd.DataFrame
        Raw loan data
    """
    # Define only the columns we need (reduces memory significantly)
    required_columns = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
        'purpose', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc', 'loan_status'
    ]
    
    # Define optimized data types to reduce memory footprint
    dtype_mapping = {
        'loan_amnt': 'float32',
        'int_rate': 'float32',
        'installment': 'float32',
        'annual_inc': 'float32',
        'dti': 'float32',
        'delinq_2yrs': 'float32',
        'open_acc': 'float32',
        'pub_rec': 'float32',
        'revol_bal': 'float32',
        'revol_util': 'float32',
        'total_acc': 'float32',
        # String columns will be loaded as object (default)
        'term': 'category',
        'grade': 'category',
        'sub_grade': 'category',
        'home_ownership': 'category',
        'verification_status': 'category',
        'purpose': 'category',
        'loan_status': 'category'
    }
    
    print(f"Loading data from {filepath}...")
    print(f"  Strategy: Chunked reading with optimized dtypes")
    
    try:
        # First, try to peek at the columns available in the file
        sample_df = pd.read_csv(filepath, nrows=5, low_memory=False)
        available_columns = [col for col in required_columns if col in sample_df.columns]
        available_dtypes = {k: v for k, v in dtype_mapping.items() if k in available_columns}
        
        print(f"  Found {len(available_columns)}/{len(required_columns)} required columns")
        
        # Determine chunk size based on available memory (smaller chunks = less memory)
        chunk_size = 100_000  # 100k rows per chunk
        
        # Read in chunks and concatenate
        chunks = []
        total_rows = 0
        
        for i, chunk in enumerate(pd.read_csv(
            filepath,
            usecols=available_columns,
            dtype=available_dtypes,
            chunksize=chunk_size,
            low_memory=False,
            on_bad_lines='skip'  # Skip problematic rows
        )):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            # Print progress every 10 chunks
            if (i + 1) % 10 == 0:
                print(f"    Processed {total_rows:,} rows...")
            
            # If sample_size is specified, stop when we have enough rows
            if sample_size and total_rows >= sample_size:
                print(f"  Reached sample size limit of {sample_size:,} rows")
                break
        
        # Concatenate all chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Apply sample size limit if needed
        if sample_size and len(df) > sample_size:
            df = df.head(sample_size)
        
        # Report memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"Loaded {len(df):,} records with {len(df.columns)} columns")
        print(f"  Memory usage: {memory_mb:.2f} MB")
        
        return df
        
    except Exception as e:
        print(f"Error during chunked reading: {e}")
        print("Falling back to basic reading with nrows limit...")
        
        # Fallback: Read only first N rows if chunked reading fails
        fallback_rows = sample_size if sample_size else 500_000
        df = pd.read_csv(
            filepath,
            nrows=fallback_rows,
            low_memory=False,
            on_bad_lines='skip'
        )
        print(f"Loaded {len(df):,} records (limited due to memory constraints)")
        return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features for credit risk modeling.
    
    We focus on features available at loan origination (not post-hoc outcomes).
    """
    # Core features commonly available in LendingClub data
    feature_cols = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
        'purpose', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc', 'loan_status'
    ]
    
    # Keep only columns that exist in the dataset
    available_cols = [col for col in feature_cols if col in df.columns]
    df_selected = df[available_cols].copy()
    
    print(f"Selected {len(available_cols)} features for modeling")
    return df_selected


def create_target(df: pd.DataFrame, target_col: str = 'loan_status') -> pd.DataFrame:
    """
    Create binary target variable: 1 = Default/Charged Off, 0 = Fully Paid.
    
    WHY BINARY?
    Regulatory frameworks (Basel III, IFRS 9) require clear default definitions.
    We classify as default: Charged Off, Default, Late (31-120 days).
    """
    default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)']
    df = df.copy()
    df['target'] = df[target_col].apply(lambda x: 1 if x in default_statuses else 0)
    
    # Remove ambiguous statuses (current loans, grace period)
    clear_statuses = ['Fully Paid', 'Charged Off', 'Default', 'Late (31-120 days)']
    df = df[df[target_col].isin(clear_statuses)]
    
    print(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values with domain-aware logic.
    
    RATIONALE:
    - Numerical: Median (robust to income/debt outliers)
    - Categorical: "Missing" category (preserves information about data availability)
    - Employment length: "Unknown" (self-employed or unreported)
    """
    df = df.copy()
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Impute numerics with median
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Imputed {col} with median: {median_val:.2f}")
    
    # Impute categoricals with "Missing"
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna('Missing', inplace=True)
            print(f"  Imputed {col} with 'Missing'")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create advanced financial features for credit risk assessment.
    
    FEATURES ENGINEERED:
    1. debt_to_income_ratio: Loan amount relative to annual income
    2. credit_utilization: Revolving balance / implied credit limit
    3. stability_index: Weighted combination of employment + housing stability
    """
    df = df.copy()
    
    # 1. DEBT-TO-INCOME RATIO
    # Higher ratio = more financial strain = higher risk
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['debt_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)  # +1 to avoid division by zero
        print("  Created: debt_to_income_ratio")
    
    # 2. CREDIT UTILIZATION
    # Industry standard: >30% utilization = higher risk
    if 'revol_util' in df.columns:
        # revol_util might be string with '%' or numeric
        if df['revol_util'].dtype == 'object':
            df['credit_utilization'] = df['revol_util'].str.rstrip('%').astype(float) / 100
        else:
            df['credit_utilization'] = df['revol_util'] / 100 if df['revol_util'].max() > 1 else df['revol_util']
        print("  Created: credit_utilization")
    
    # 3. STABILITY INDEX
    # Combines employment length and home ownership into a single stability score
    # Employment length parsing
    if 'emp_length' in df.columns:
        def parse_emp_length(val):
            if pd.isna(val) or val == 'Missing':
                return 0
            val = str(val).lower()
            if '10+' in val:
                return 10
            elif '< 1' in val or '<1' in val:
                return 0.5
            else:
                try:
                    return float(''.join(filter(str.isdigit, val.split()[0])))
                except:
                    return 0
        
        df['emp_length_years'] = df['emp_length'].apply(parse_emp_length)
        emp_normalized = df['emp_length_years'] / 10  # Normalize to 0-1
    else:
        emp_normalized = 0.5  # Default if not available
    
    # Home ownership scoring
    if 'home_ownership' in df.columns:
        ownership_scores = {'OWN': 1.0, 'MORTGAGE': 0.7, 'RENT': 0.3, 'OTHER': 0.2, 'NONE': 0.1, 'Missing': 0.2}
        df['home_score'] = df['home_ownership'].map(ownership_scores).fillna(0.2)
    else:
        df['home_score'] = 0.5
    
    # Combine into stability index
    # WHY 0.6/0.4 WEIGHTS? Employment length is slightly more predictive per literature
    df['stability_index'] = 0.6 * emp_normalized + 0.4 * df['home_score']
    print("  Created: stability_index")
    
    # Clean up intermediate columns
    cols_to_drop = ['emp_length_years', 'home_score']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    return df


def encode_categoricals(df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical variables for XGBoost.
    
    Uses Label Encoding (XGBoost handles this natively).
    Returns encoders dict for later inference.
    """
    df = df.copy()
    encoders = {}
    
    # Include both 'object' and 'category' dtypes (category is used for memory optimization)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Exclude target if it's in there
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  Encoded: {col} ({len(le.classes_)} categories)")
    
    return df, encoders


def split_data(df: pd.DataFrame, target_col: str = 'target', 
               test_size: float = 0.30, random_state: int = 42) -> Tuple:
    """
    Stratified train-test split (70/30).
    
    WHY STRATIFIED?
    Preserves the default rate distribution in both splits, critical for 
    imbalanced credit data.
    """
    X = df.drop(columns=[target_col, 'loan_status'], errors='ignore')
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")
    print(f"Train default rate: {y_train.mean():.2%} | Test default rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, 
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance the training data.
    
    WHY SMOTE?
    - Synthetic Minority Over-sampling TEchnique creates synthetic examples
    - Better than random oversampling (which just duplicates)
    - Better than undersampling (which loses information)
    - Regulatory-friendly: preserves all real default cases
    """
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def run_preprocessing_pipeline(filepath: str) -> Tuple:
    """
    Execute the full preprocessing pipeline.
    
    Returns: X_train, X_test, y_train, y_test, encoders, feature_names
    """
    print("=" * 60)
    print("CREDIT RISK PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/7] Loading data...")
    df = load_data(filepath)
    
    # Step 2: Select features
    print("\n[2/7] Selecting features...")
    df = select_features(df)
    
    # Step 3: Create target
    print("\n[3/7] Creating target variable...")
    df = create_target(df)
    
    # Step 4: Handle missing values
    print("\n[4/7] Handling missing values...")
    df = handle_missing_values(df)
    
    # Step 5: Engineer features
    print("\n[5/7] Engineering features...")
    df = engineer_features(df)
    
    # Step 6: Encode categoricals
    print("\n[6/7] Encoding categorical variables...")
    df, encoders = encode_categoricals(df)
    
    # Step 7: Split and balance
    print("\n[7/7] Splitting and balancing data...")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    
    feature_names = X_train.columns.tolist()
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"Features: {len(feature_names)}")
    print("=" * 60)
    
    return X_train_balanced, X_test, y_train_balanced, y_test, encoders, feature_names


if __name__ == "__main__":
    # Example usage
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "loan.csv")
    if os.path.exists(data_path):
        X_train, X_test, y_train, y_test, encoders, features = run_preprocessing_pipeline(data_path)
    else:
        print(f"Dataset not found at {data_path}")
        print("Please place loan.csv in the data/ folder")
