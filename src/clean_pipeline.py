"""
clean_pipeline.py
End-to-End Data Cleaning Pipeline for Week 2 Task 1: Data Cleaning Project.
Implements the full 10-step data cleaning lifecycle:
1. Load dataset & initial exploration
2. Visualize missing data with Heatmap & Bar charts
3. Apply column-specific missing data imputation strategies
4. Detect and drop duplicate rows (exact & identifier level)
5. Standardize column names to snake_case and enforce correct data types
6. Detect and handle outliers using IQR and Z-Score methods
7. Categorical encoding (One-Hot, Ordinal, and Binary) & Normalization
8. Validate cleaned dataset with summary statistics & assertions
9. Export cleaned datasets to CSV
10. Generate before-and-after data quality comparison tables
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_cleaning_pipeline(raw_path: str, output_dir: str):
    # Setup directory structure
    cleaned_dir = os.path.join(output_dir, "data", "cleaned")
    figures_dir = os.path.join(output_dir, "outputs", "figures")
    tables_dir = os.path.join(output_dir, "outputs", "tables")
    os.makedirs(cleaned_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 80)
    print("STEP 1: LOAD DATASET & INITIAL EXPLORATION")
    print("=" * 80)
    df_raw = pd.read_csv(raw_path)
    initial_shape = df_raw.shape
    initial_null_count = df_raw.isnull().sum().sum()
    initial_exact_duplicates = df_raw.duplicated().sum()
    print(f"Loaded raw dataset from: {raw_path}")
    print(f"Initial Shape: {initial_shape[0]} rows, {initial_shape[1]} columns")
    print(f"Total Missing Values: {initial_null_count} cells")
    print(f"Total Exact Duplicate Rows: {initial_exact_duplicates}")
    print("\nInitial Column Names & Types:")
    for col in df_raw.columns:
        print(f" - '{col}': {df_raw[col].dtype} ({df_raw[col].isnull().sum()} nulls)")

    print("\n" + "=" * 80)
    print("STEP 2: VISUALIZE MISSING DATA")
    print("=" * 80)
    
    # 2.1 Missing Data Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_raw.isnull(), cbar=True, cmap="YlOrRd", yticklabels=False)
    plt.title("Missing Data Heatmap (Raw Dataset Before Cleaning)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Columns / Features", fontsize=11, labelpad=8)
    plt.ylabel("Row Index (White/Red = Missing)", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    heatmap_path = os.path.join(figures_dir, "01_missing_data_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved missing data heatmap to: {heatmap_path}")

    # 2.2 Missing Data Bar Chart
    missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(missing_pct.index, missing_pct.values, color="#E11D48", edgecolor="#9F1239", alpha=0.85)
    plt.title("Percentage of Missing Values by Column (Raw Data)", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Missing Percentage (%)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylim(0, max(missing_pct.values) * 1.25)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    barchart_path = os.path.join(figures_dir, "02_missing_data_bar.png")
    plt.savefig(barchart_path, dpi=300)
    plt.close()
    print(f"Saved missing data bar chart to: {barchart_path}")

    df = df_raw.copy()

    print("\n" + "=" * 80)
    print("STEP 4: DETECT & REMOVE DUPLICATE ROWS")
    print("=" * 80)
    # Detect exact duplicates
    exact_dups = df.duplicated()
    print(f"Detected {exact_dups.sum()} exact duplicate rows.")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape after dropping exact duplicates: {df.shape}")

    print("\n" + "=" * 80)
    print("STEP 5: STANDARDIZE COLUMN NAMES TO snake_case")
    print("=" * 80)
    def clean_column_name(col: str) -> str:
        # Strip whitespace, lowercase, remove special characters, replace spaces/hyphens with underscore
        col = col.strip().lower()
        col = re.sub(r'[\$\(\)]', '', col)
        col = re.sub(r'[\s\-]+', '_', col)
        col = re.sub(r'_+', '_', col)
        col = col.strip('_')
        # Specific mappings for clarity
        mapping = {
            "customer_id": "customer_id",
            "full_name": "full_name",
            "age": "age",
            "gender": "gender",
            "annual_income": "annual_income",
            "tenure_months": "tenure_months",
            "contract_type": "contract_type",
            "payment_method": "payment_method",
            "monthly_charges": "monthly_charges",
            "total_charges": "total_charges",
            "internet_service": "internet_service",
            "tech_support": "tech_support",
            "satisfaction_score_1_5": "satisfaction_score",
            "satisfaction_score": "satisfaction_score",
            "join_date": "join_date",
            "city": "city",
            "churn": "churn"
        }
        return mapping.get(col, col)

    df.columns = [clean_column_name(c) for c in df.columns]
    print(f"Standardized column names: {list(df.columns)}")

    # Secondary duplicate check on primary key 'customer_id'
    # Drop rows where customer_id is null first (unidentifiable transactions)
    null_cust_ids = df['customer_id'].isnull().sum()
    print(f"Rows with null customer_id: {null_cust_ids} -> Dropping unidentifiable records.")
    df = df.dropna(subset=['customer_id']).reset_index(drop=True)

    pk_dups = df.duplicated(subset=['customer_id'], keep='first').sum()
    print(f"Detected {pk_dups} duplicate customer_id records. Retaining first occurrence.")
    df = df.drop_duplicates(subset=['customer_id'], keep='first').reset_index(drop=True)
    print(f"Shape after primary key deduplication: {df.shape}")

    # Also drop records where target variable 'churn' is null (to avoid injecting target bias in analytical modeling)
    null_churn = df['churn'].isnull().sum()
    print(f"Rows with missing target 'churn': {null_churn} -> Dropping unlabelled targets.")
    df = df.dropna(subset=['churn']).reset_index(drop=True)
    print(f"Shape after target validation: {df.shape}")

    print("\n" + "=" * 80)
    print("STEP 3 & 5 (CONT): COLUMN-SPECIFIC CLEANING & DATA TYPE CORRECTIONS")
    print("=" * 80)

    # 1. full_name
    df['full_name'] = df['full_name'].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip()).title())

    # 2. annual_income ($) -> Clean currency string to float, handle negatives, impute median
    def parse_currency(val):
        if pd.isnull(val):
            return np.nan
        s = str(val).strip()
        is_neg = '-' in s
        s = re.sub(r'[\$,\s\-]', '', s)
        try:
            num = float(s)
            return -num if is_neg else num
        except ValueError:
            return np.nan

    df['annual_income'] = df['annual_income'].apply(parse_currency)
    # Erroneous negative incomes treated as invalid -> replace with NaN
    neg_incomes = (df['annual_income'] < 0).sum()
    if neg_incomes > 0:
        print(f"Found {neg_incomes} negative annual_income records -> Setting to NaN.")
        df.loc[df['annual_income'] < 0, 'annual_income'] = np.nan
    median_income = df['annual_income'].median()
    df['annual_income'] = df['annual_income'].fillna(median_income)
    print(f"Imputed missing annual_income with median: ${median_income:,.2f}")

    # 3. age -> Clean negative entries, out-of-range errors, impute median, cast int64
    # Negative age entries are entry typos (e.g. -5 -> 5 or replace with NaN)
    df.loc[df['age'] < 0, 'age'] = np.nan
    df.loc[df['age'] > 120, 'age'] = np.nan  # impossible human age
    median_age = df['age'].median()
    df['age'] = df['age'].fillna(median_age).round().astype(int)
    print(f"Cleaned age: imputed missing with median ({median_age:.0f} yrs) and cast to int64.")

    # 4. tenure_months -> Clean negative entries, round, impute median, cast int64
    df.loc[df['tenure_months'] < 0, 'tenure_months'] = 0
    median_tenure = df['tenure_months'].median()
    df['tenure_months'] = df['tenure_months'].fillna(median_tenure).round().astype(int)
    print(f"Cleaned tenure_months: imputed missing with median ({median_tenure:.0f} mos) and cast to int64.")

    # 5. monthly_charges -> Clean zero or negative values, impute median, cast float64
    df.loc[df['monthly_charges'] <= 0, 'monthly_charges'] = np.nan
    median_mc = df['monthly_charges'].median()
    df['monthly_charges'] = df['monthly_charges'].fillna(median_mc).astype(float)
    print(f"Cleaned monthly_charges: imputed missing with median (${median_mc:.2f}).")

    # 6. total_charges -> Handle empty blanks " ", currency strings, impute via tenure * monthly_charges
    def parse_total_charges(val):
        if pd.isnull(val):
            return np.nan
        s = str(val).strip()
        if s == "" or s.lower() in ["nan", "null", "none", " "]:
            return np.nan
        s = re.sub(r'[\$,\s]', '', s)
        try:
            return float(s)
        except ValueError:
            return np.nan

    df['total_charges'] = df['total_charges'].apply(parse_total_charges)
    # Recalculate / interpolate missing total charges using tenure_months * monthly_charges
    missing_tc = df['total_charges'].isnull().sum()
    print(f"Reconciling {missing_tc} missing total_charges records via tenure_months * monthly_charges...")
    df['total_charges'] = df['total_charges'].fillna(df['tenure_months'] * df['monthly_charges'])

    # 7. satisfaction_score -> Clean values outside [1, 5], impute median (3), cast int64
    df.loc[~df['satisfaction_score'].isin([1, 2, 3, 4, 5]), 'satisfaction_score'] = np.nan
    median_score = df['satisfaction_score'].median()
    df['satisfaction_score'] = df['satisfaction_score'].fillna(median_score).round().astype(int)
    print(f"Cleaned satisfaction_score: cast to int64 (scale 1-5).")

    # 8. join_date -> Convert mixed date formats, coerce errors, impute missing dates
    df['join_date'] = pd.to_datetime(df['join_date'], format='mixed', errors='coerce')
    # Impute missing dates using estimated join date: reference_date (2024-12-31) - tenure_months
    ref_date = pd.Timestamp("2024-12-31")
    missing_dates = df['join_date'].isnull().sum()
    if missing_dates > 0:
        imputed_dates = ref_date - pd.to_timedelta(df.loc[df['join_date'].isnull(), 'tenure_months'] * 30.4375, unit='D')
        df.loc[df['join_date'].isnull(), 'join_date'] = imputed_dates
    print(f"Cleaned join_date: parsed to datetime64[ns] ({missing_dates} imputed from tenure).")

    # 9. Categorical variables standardization
    # gender
    def clean_gender(val):
        if pd.isnull(val):
            return "Unspecified"
        s = str(val).strip().lower()
        if s in ["male", "m"]:
            return "Male"
        elif s in ["female", "f"]:
            return "Female"
        else:
            return "Other"
    df['gender'] = df['gender'].apply(clean_gender)

    # contract_type
    def clean_contract(val):
        if pd.isnull(val):
            return "Month-to-Month"
        s = str(val).strip().lower()
        if "two" in s or "2" in s:
            return "Two Year"
        elif "one" in s or "1" in s:
            return "One Year"
        else:
            return "Month-to-Month"
    df['contract_type'] = df['contract_type'].apply(clean_contract)

    # payment_method
    def clean_payment(val):
        if pd.isnull(val):
            return "Unknown"
        s = str(val).strip().lower()
        if "electronic" in s or "e-check" in s:
            return "Electronic Check"
        elif "mailed" in s or "check" in s:
            return "Mailed Check"
        elif "bank" in s:
            return "Bank Transfer"
        elif "credit" in s:
            return "Credit Card"
        else:
            return "Unknown"
    df['payment_method'] = df['payment_method'].apply(clean_payment)

    # internet_service
    def clean_internet(val):
        if pd.isnull(val):
            return "No"
        s = str(val).strip().lower()
        if "fiber" in s:
            return "Fiber Optic"
        elif "dsl" in s:
            return "DSL"
        else:
            return "No"
    df['internet_service'] = df['internet_service'].apply(clean_internet)

    # tech_support
    def clean_tech_support(val):
        if pd.isnull(val):
            return "No"
        s = str(val).strip().lower()
        if s in ["yes", "y", "true", "1"]:
            return "Yes"
        elif "no internet" in s:
            return "No internet service"
        else:
            return "No"
    df['tech_support'] = df['tech_support'].apply(clean_tech_support)

    # city
    df['city'] = df['city'].astype(str).apply(lambda x: "Unknown" if x.strip().lower() in ["nan", "null", "none", ""] else x.strip().title())

    # churn
    def clean_churn(val):
        s = str(val).strip().lower()
        if s in ["yes", "y", "true", "1"]:
            return "Yes"
        else:
            return "No"
    df['churn'] = df['churn'].apply(clean_churn)

    print("\n" + "=" * 80)
    print("STEP 6: OUTLIER DETECTION & TREATMENT (IQR & Z-SCORE METHODS)")
    print("=" * 80)
    num_cols = ["age", "annual_income", "tenure_months", "monthly_charges", "total_charges"]
    
    # 6.1 Visualize Boxplots BEFORE Outlier Capping
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col], ax=axes[i], color="#38BDF8", flierprops={"marker": "o", "color": "#E11D48", "markersize": 5})
        axes[i].set_title(col.replace('_', ' ').title(), fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.6)
    plt.suptitle("Outlier Inspection via Boxplots (Before Treatment)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    outlier_before_path = os.path.join(figures_dir, "03_outliers_before_boxplots.png")
    plt.savefig(outlier_before_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pre-treatment boxplots to: {outlier_before_path}")

    # 6.2 Compute IQR and Z-Score Statistics
    outlier_stats = []
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        mean_val = df[col].mean()
        std_val = df[col].std()
        z_scores = np.abs((df[col] - mean_val) / std_val)
        z_outliers = (z_scores > 3.0).sum()

        outlier_stats.append({
            "feature": col,
            "mean": mean_val,
            "std": std_val,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "iqr_lower": lower_bound,
            "iqr_upper": upper_bound,
            "iqr_outlier_count": iqr_outliers,
            "z_outlier_count": z_outliers
        })
        print(f"[{col}] IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}] -> {iqr_outliers} outliers | Z-score (|Z|>3): {z_outliers} outliers")

    # 6.3 Handle Outliers via Winsorization / IQR Capping for skewed financials
    # annual_income and monthly_charges have severe unphysical or extreme billing outliers
    for col in ["annual_income", "monthly_charges"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = max(0.0, q1 - 1.5 * iqr)
        upper = q3 + 1.5 * iqr
        initial_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"Capped {initial_clipped} extreme values in '{col}' to range [{lower:.2f}, {upper:.2f}].")

    # Re-sync total_charges if tenure and monthly charges changed
    df['total_charges'] = df['total_charges'].clip(lower=0.0, upper=df['tenure_months'].max() * df['monthly_charges'].max())

    # 6.4 Visualize Boxplots AFTER Outlier Capping
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col], ax=axes[i], color="#34D399", flierprops={"marker": "o", "color": "#059669", "markersize": 4})
        axes[i].set_title(col.replace('_', ' ').title(), fontsize=11, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.6)
    plt.suptitle("Outlier Inspection via Boxplots (After IQR Capping Treatment)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    outlier_after_path = os.path.join(figures_dir, "04_outliers_after_boxplots.png")
    plt.savefig(outlier_after_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved post-treatment boxplots to: {outlier_after_path}")

    # 6.5 Visualize Distributions of Cleaned Numerical Features
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#2563EB", bins=25, edgecolor="#1E40AF")
        axes[i].set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col.replace('_', ' ').title(), fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
    # 6th plot: Satisfaction score countplot
    sns.countplot(x=df['satisfaction_score'], ax=axes[5], palette="Blues_d", hue=df['satisfaction_score'], legend=False)
    axes[5].set_title("Satisfaction Score Distribution (1-5)", fontsize=11, fontweight="bold")
    axes[5].set_xlabel("Satisfaction Score", fontsize=10)
    axes[5].set_ylabel("Count", fontsize=10)
    axes[5].grid(axis="y", linestyle="--", alpha=0.5)
    plt.suptitle("Post-Cleaning Distributions of Numerical & Ordinal Attributes", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    dist_path = os.path.join(figures_dir, "05_numerical_distributions.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved numerical distributions plot to: {dist_path}")

    print("\n" + "=" * 80)
    print("STEP 7: CATEGORICAL ENCODING & SCALING")
    print("=" * 80)
    # Create ML-ready encoded copy
    df_encoded = df.copy()

    # 7.1 Ordinal Encoding for contract_type (Month-to-Month < One Year < Two Year)
    contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    df_encoded['contract_type_encoded'] = df_encoded['contract_type'].map(contract_map)

    # 7.2 Binary Encoding for target churn (No: 0, Yes: 1)
    churn_map = {"No": 0, "Yes": 1}
    df_encoded['churn_encoded'] = df_encoded['churn'].map(churn_map)

    # 7.3 One-Hot Encoding for nominal variables
    nominal_cols = ["gender", "internet_service", "tech_support", "payment_method"]
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_cols, drop_first=True, dtype=int)
    print(f"Applied One-Hot Encoding on: {nominal_cols}. Encoded dataset shape: {df_encoded.shape}")

    # 7.4 Min-Max Normalization example for ML features
    for col in ["annual_income", "monthly_charges", "total_charges"]:
        c_min = df_encoded[col].min()
        c_max = df_encoded[col].max()
        df_encoded[f"{col}_normalized"] = (df_encoded[col] - c_min) / (c_max - c_min)
    print("Created normalized features [0, 1] for annual_income, monthly_charges, and total_charges.")

    print("\n" + "=" * 80)
    print("STEP 8: VALIDATE CLEANED DATASET WITH SUMMARY STATISTICS & ASSERTIONS")
    print("=" * 80)
    final_shape = df.shape
    final_nulls = df.isnull().sum().sum()
    final_exact_dups = df.duplicated().sum()
    final_pk_dups = df.duplicated(subset=['customer_id']).sum()

    assert final_nulls == 0, f"Error: Found {final_nulls} nulls in cleaned dataframe!"
    assert final_exact_dups == 0, f"Error: Found {final_exact_dups} duplicate rows in cleaned dataframe!"
    assert final_pk_dups == 0, f"Error: Found {final_pk_dups} duplicate primary keys in cleaned dataframe!"
    assert (df['age'] < 18).sum() == 0, "Error: Age below minimum adult threshold!"
    assert (df['annual_income'] < 0).sum() == 0, "Error: Negative income detected!"
    assert (df['monthly_charges'] <= 0).sum() == 0, "Error: Non-positive monthly charges detected!"

    print("ALL DATA INTEGRITY ASSERTIONS PASSED:")
    print(f" - Cleaned Record Count: {final_shape[0]} rows")
    print(f" - Cleaned Feature Count: {final_shape[1]} columns")
    print(f" - Missing Values: {final_nulls} (100% complete)")
    print(f" - Exact Duplicate Rows: {final_exact_dups}")
    print(f" - Customer ID Uniqueness: 100% unique ({df['customer_id'].nunique()} distinct IDs)")

    print("\n" + "=" * 80)
    print("STEP 9: EXPORT CLEANED DATASETS")
    print("=" * 80)
    cleaned_csv_path = os.path.join(cleaned_dir, "customer_churn_cleaned.csv")
    encoded_csv_path = os.path.join(cleaned_dir, "customer_churn_encoded.csv")
    
    df.to_csv(cleaned_csv_path, index=False)
    df_encoded.to_csv(encoded_csv_path, index=False)
    print(f"Saved cleaned analysis-ready dataset to: {cleaned_csv_path}")
    print(f"Saved ML-ready encoded dataset to: {encoded_csv_path}")

    print("\n" + "=" * 80)
    print("STEP 10: BEFORE-AND-AFTER COMPARISON TABLE")
    print("=" * 80)
    comparison_data = [
        {"Metric": "Total Rows", "Before Cleaning (Raw)": f"{initial_shape[0]:,}", "After Cleaning": f"{final_shape[0]:,}", "Delta / Impact": f"-{initial_shape[0] - final_shape[0]} rows removed"},
        {"Metric": "Total Columns", "Before Cleaning (Raw)": str(initial_shape[1]), "After Cleaning": str(final_shape[1]), "Delta / Impact": "Standardized to clean snake_case"},
        {"Metric": "Total Missing Values", "Before Cleaning (Raw)": f"{initial_null_count:,} cells ({initial_null_count / (initial_shape[0]*initial_shape[1]):.1%})", "After Cleaning": "0 cells (0.0%)", "Delta / Impact": "100% imputed / resolved"},
        {"Metric": "Exact Duplicate Rows", "Before Cleaning (Raw)": str(initial_exact_duplicates), "After Cleaning": "0", "Delta / Impact": f"{initial_exact_duplicates} duplicates purged"},
        {"Metric": "Duplicate Customer IDs", "Before Cleaning (Raw)": f"{df_raw.duplicated(subset=[' Customer_ID ']).sum()}", "After Cleaning": "0", "Delta / Impact": "Ensured 1-to-1 customer granularity"},
        {"Metric": "Column Naming Standard", "Before Cleaning (Raw)": "Messy (' Customer_ID ', 'Annual Income ($)')", "After Cleaning": "Normalized snake_case", "Delta / Impact": "Programmatic consistency"},
        {"Metric": "Annual Income Data Type", "Before Cleaning (Raw)": "object (strings with '$', ',')", "After Cleaning": "float64 (numeric)", "Delta / Impact": "Enables mathematical aggregation"},
        {"Metric": "Total Charges Data Type", "Before Cleaning (Raw)": "object (strings with ' ' blanks)", "After Cleaning": "float64 (numeric)", "Delta / Impact": "Reconciled with tenure * monthly"},
        {"Metric": "Join Date Data Type", "Before Cleaning (Raw)": "object (mixed date string formats)", "After Cleaning": "datetime64[ns]", "Delta / Impact": "Enables temporal time-series queries"},
        {"Metric": "Negative Age / Incomes", "Before Cleaning (Raw)": "Present (entry typos)", "After Cleaning": "0 (purged / imputed)", "Delta / Impact": "Domain logic enforced"},
        {"Metric": "Extreme Outliers ($999/mo, $3M/yr)", "Before Cleaning (Raw)": "Severe skewness present", "After Cleaning": "Winsorized / IQR capped", "Delta / Impact": "Robust statistical distributions"},
        {"Metric": "Categorical Integrity", "Before Cleaning (Raw)": "Inconsistent casing ('m', 'Male', '1')", "After Cleaning": "Standardized canonical taxonomy", "Delta / Impact": "No fragmented factor levels"}
    ]
    comparison_df = pd.DataFrame(comparison_data)
    comparison_csv_path = os.path.join(tables_dir, "before_after_metrics.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"Saved before-and-after metrics table to: {comparison_csv_path}")
    print(comparison_df.to_string(index=False))

    return df, df_encoded, comparison_df

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_csv = os.path.join(base_dir, "data", "raw", "customer_churn_raw.csv")
    run_cleaning_pipeline(raw_csv, base_dir)
