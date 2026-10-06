"""
build_notebook.py
Builds and executes the comprehensive Jupyter Notebook:
notebooks/data_cleaning_project.ipynb
Covering all 10 implementation steps, key requirements, rich markdown explanations,
clean code cells, and executing all cells using nbclient so that all outputs
(tables, plots, stats) are pre-rendered.
"""

import os
# pyright: ignore[reportMissingImports]
import nbformat as nbf
# pyright: ignore[reportMissingImports]
from nbclient import NotebookClient

def create_and_execute_notebook(notebook_path: str):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.4"
        }
    }

    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Task 1: End-to-End Data Cleaning Project
**Organization**: WeIntern Pvt Ltd  
**Track**: Data Science & Machine Learning Engineering  
**Module**: Advanced Data Preparation & Quality Assurance  
**Author**: Bineeta Yadav (Data Science Intern, WeIntern Pvt Ltd)  
**Repository**: [Task1_Data_Cleaning_Project](https://github.com/bineetayadav/Task1_Data_Cleaning_Project)  
**Dataset**: Enterprise Customer Churn & Demographics (`customer_churn_raw.csv`)  
**Objective**: Ingest a messy, raw real-world dataset and transform it into a pristine, analysis-ready and ML-ready format using Python's Pandas, NumPy, Matplotlib, and Seaborn.

---

### Project Lifecycle & Implementation Workflow
This notebook walks through the systematic 10-step data cleaning lifecycle:
1. **Initial Exploration & Profiling**: Ingest raw dataset; audit shape, schema, `.info()`, `.describe()`, and initial samples.
2. **Missing Data Diagnostics**: Visualize missingness patterns using Seaborn heatmaps and column-wise percentage bars.
3. **Missing Value Strategy & Imputation**: Formulate and apply column-specific strategies (drop unidentifiable records, median imputation for skewed numericals, mode/unknown imputation for categoricals, tenure-reconciled totals).
4. **Duplicate Row Auditing & Deduplication**: Detect exact duplicates and primary key conflicts using `.duplicated()`, remove with `.drop_duplicates()`.
5. **Column Standardization & Type Corrections**: Normalize column names to `snake_case`; correct string currencies, blanks, and mixed date strings into proper numeric and `datetime64[ns]` formats.
6. **Outlier Detection & Statistical Handling**: Identify anomalies using Boxplots, IQR method, and Z-score evaluation ($|Z| > 3$); apply Winsorization/clipping.
7. **Categorical Encoding & Feature Scaling**: Implement One-Hot Encoding for nominal variables, Ordinal Encoding for ranked tiers, and Min-Max feature normalization.
8. **Post-Cleaning Validation**: Validate integrity using descriptive statistics, zero-null assertions, and distribution plots.
9. **Export Cleaned Assets**: Export cleaned analysis-ready CSV and ML-encoded CSV.
10. **Data Quality Evaluation**: Before-and-after comparison table of comprehensive data quality metrics.
"""))

    # Imports & Setup Cell
    cells.append(nbf.v4.new_markdown_cell("""## Step 0: Environment Setup & Library Imports
We initialize standard data manipulation and visualization libraries with configured styling parameters.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure presentation styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['figure.facecolor'] = '#FFFFFF'
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

print("Environment configured successfully. Ready for data cleaning pipeline.")"""))

    # Step 1: Load and Initial Exploration
    cells.append(nbf.v4.new_markdown_cell("""## Step 1: Load Dataset & Perform Initial Exploration
We load the raw, messy dataset from `../data/raw/customer_churn_raw.csv` and inspect its dimensions, data types, missing values count, and top rows.
"""))

    cells.append(nbf.v4.new_code_cell("""# Load raw dataset
raw_df = pd.read_csv('../data/raw/customer_churn_raw.csv')

print(f"Raw Dataset Shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
print(f"Total Cells: {raw_df.size:,}")
print(f"Total Missing Cells: {raw_df.isnull().sum().sum():,} ({raw_df.isnull().sum().sum() / raw_df.size:.2%})")
print(f"Total Exact Duplicate Rows: {raw_df.duplicated().sum()}")"""))

    cells.append(nbf.v4.new_code_cell("""# Display schema information and datatypes
raw_df.info()"""))

    cells.append(nbf.v4.new_code_cell("""# Preview first 5 records of raw dataset
raw_df.head()"""))

    cells.append(nbf.v4.new_code_cell("""# Preliminary summary statistics for numerical attributes
raw_df.describe().T"""))

    # Step 2: Visualize Missing Data
    cells.append(nbf.v4.new_markdown_cell("""## Step 2: Visualize Missing Data
To diagnose whether missingness occurs at random or across specific features, we construct a Missing Data Heatmap using Seaborn alongside a column-wise percentage bar chart.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.1 Missing Data Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(raw_df.isnull(), cbar=True, cmap="YlOrRd", yticklabels=False)
plt.title("Missing Data Heatmap (Raw Dataset Before Cleaning)", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Columns / Features", fontsize=11, labelpad=8)
plt.ylabel("Row Index (Red/Yellow = Missing)", fontsize=11, labelpad=8)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 2.2 Missing Values Breakdown by Column
null_counts = raw_df.isnull().sum()
null_percent = (null_counts / len(raw_df)) * 100
missing_summary = pd.DataFrame({
    'Missing_Count': null_counts,
    'Missing_Percentage': null_percent
}).loc[lambda df: df['Missing_Count'] > 0].sort_values(by='Missing_Percentage', ascending=False)

print("Columns with Missing Values:")
display(missing_summary)

plt.figure(figsize=(10, 5))
bars = plt.bar(missing_summary.index, missing_summary['Missing_Percentage'], color='#E11D48', edgecolor='#9F1239', alpha=0.85)
plt.title("Percentage of Missing Values by Column (Raw Data)", fontsize=13, fontweight="bold", pad=12)
plt.ylabel("Missing Percentage (%)", fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.ylim(0, missing_summary['Missing_Percentage'].max() * 1.25)
for bar in bars:
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()"""))

    # Step 4: Duplicate Detection & Removal
    cells.append(nbf.v4.new_markdown_cell("""## Step 4: Detect & Remove Duplicate Rows
We identify and eliminate both:
1. **Exact duplicate rows** where all columns are identical due to repeated batch ingestion.
2. **Primary Key duplicates** where the same `Customer_ID` was logged multiple times with slight variations.
"""))

    cells.append(nbf.v4.new_code_cell("""df = raw_df.copy()

# Audit exact duplicates
exact_dups = df.duplicated()
print(f"Exact duplicate rows detected: {exact_dups.sum()}")

# Drop exact duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dataset shape after dropping exact duplicates: {df.shape}")"""))

    # Step 5: Standardize Column Names
    cells.append(nbf.v4.new_markdown_cell("""## Step 5: Standardize Column Names to `snake_case`
Raw column names frequently contain leading/trailing whitespaces, mixed casing, parentheses, dollar signs, and hyphens (e.g. `' Customer_ID '`, `'Annual Income ($)'`, `'Contract-Type'`). We normalize all headers to clean `snake_case`.
"""))

    cells.append(nbf.v4.new_code_cell("""def clean_column_name(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r'[\$\(\)]', '', col)
    col = re.sub(r'[\s\-]+', '_', col)
    col = re.sub(r'_+', '_', col)
    col = col.strip('_')
    # Specific normalization
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
print("Standardized Column Names:")
print(list(df.columns))"""))

    cells.append(nbf.v4.new_code_cell("""# Primary key validation on 'customer_id'
# 1. Drop rows where customer_id is null (cannot identify customer)
null_ids = df['customer_id'].isnull().sum()
print(f"Rows with null customer_id: {null_ids} -> Dropping")
df = df.dropna(subset=['customer_id']).reset_index(drop=True)

# 2. Drop duplicate customer_id rows (keep first occurrence)
pk_dups = df.duplicated(subset=['customer_id'], keep='first').sum()
print(f"Duplicate customer_id entries: {pk_dups} -> Removing secondary occurrences")
df = df.drop_duplicates(subset=['customer_id'], keep='first').reset_index(drop=True)

# 3. Drop records with missing target variable 'churn' (prevent synthetic bias in analysis)
null_targets = df['churn'].isnull().sum()
print(f"Records with missing target 'churn': {null_targets} -> Dropping")
df = df.dropna(subset=['churn']).reset_index(drop=True)

print(f"Shape after identifier and target validation: {df.shape}")"""))

    # Step 3 & 5 (Cont): Column-Specific Imputation & Type Corrections
    cells.append(nbf.v4.new_markdown_cell("""## Step 3 & 5 (Continued): Column-Specific Missing Value Imputation & Type Corrections
We apply domain-appropriate strategies for each attribute:
- **`full_name`**: Clean extra spaces and standardize to Title Case.
- **`annual_income`**: Strip currency formatting (`$`, `,`), treat negative numbers as entry errors, impute missing values with the **median**, and convert to `float64`.
- **`age`**: Treat negative entries as entry typos, remove impossible human ages (>120), impute missing with **median**, and cast to integer `int64`.
- **`tenure_months`**: Replace negative tenures with 0, impute missing with **median**, and cast to integer `int64`.
- **`monthly_charges`**: Replace zeros and unphysical negative values, impute missing with **median**, and cast to `float64`.
- **`total_charges`**: Convert empty space strings (`' '`) to NaN, strip formatting, and dynamically recalculate missing total charges as $\\text{tenure\\_months} \\times \\text{monthly\\_charges}$.
- **`satisfaction_score`**: Filter invalid scale entries (<1 or >5), impute missing with median score (3), and cast to `int64`.
- **`join_date`**: Parse mixed date formats (`YYYY-MM-DD`, `DD/MM/YYYY`, `MM-DD-YYYY`) to standard `datetime64[ns]`, imputing missing dates using tenure.
- **Categoricals** (`gender`, `contract_type`, `payment_method`, `internet_service`, `tech_support`, `city`, `churn`): Normalize spelling variations and impute missing values with mode or dedicated `'Unknown'` category.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. full_name
df['full_name'] = df['full_name'].astype(str).apply(lambda x: re.sub(r'\\s+', ' ', x.strip()).title())

# 2. annual_income ($)
def parse_currency(val):
    if pd.isnull(val):
        return np.nan
    s = str(val).strip()
    is_neg = '-' in s
    s = re.sub(r'[\\$,\\s\\-]', '', s)
    try:
        num = float(s)
        return -num if is_neg else num
    except ValueError:
        return np.nan

df['annual_income'] = df['annual_income'].apply(parse_currency)
df.loc[df['annual_income'] < 0, 'annual_income'] = np.nan
median_income = df['annual_income'].median()
df['annual_income'] = df['annual_income'].fillna(median_income)
print(f"Annual Income: parsed currency strings, imputed missing with median (${median_income:,.2f})")"""))

    cells.append(nbf.v4.new_code_cell("""# 3. age
df.loc[df['age'] < 0, 'age'] = np.nan
df.loc[df['age'] > 120, 'age'] = np.nan
median_age = df['age'].median()
df['age'] = df['age'].fillna(median_age).round().astype(int)
print(f"Age: imputed missing with median ({median_age:.0f} yrs) and cast to int64")

# 4. tenure_months
df.loc[df['tenure_months'] < 0, 'tenure_months'] = 0
median_tenure = df['tenure_months'].median()
df['tenure_months'] = df['tenure_months'].fillna(median_tenure).round().astype(int)
print(f"Tenure Months: imputed missing with median ({median_tenure:.0f} mos) and cast to int64")

# 5. monthly_charges
df.loc[df['monthly_charges'] <= 0, 'monthly_charges'] = np.nan
median_mc = df['monthly_charges'].median()
df['monthly_charges'] = df['monthly_charges'].fillna(median_mc).astype(float)
print(f"Monthly Charges: imputed missing with median (${median_mc:.2f}) and cast to float64")"""))

    cells.append(nbf.v4.new_code_cell("""# 6. total_charges
def parse_total_charges(val):
    if pd.isnull(val):
        return np.nan
    s = str(val).strip()
    if s == "" or s.lower() in ["nan", "null", "none", " "]:
        return np.nan
    s = re.sub(r'[\\$,\\s]', '', s)
    try:
        return float(s)
    except ValueError:
        return np.nan

df['total_charges'] = df['total_charges'].apply(parse_total_charges)
# Impute missing total_charges via domain formula: tenure_months * monthly_charges
missing_tc = df['total_charges'].isnull().sum()
df['total_charges'] = df['total_charges'].fillna(df['tenure_months'] * df['monthly_charges'])
print(f"Total Charges: reconciled {missing_tc} missing values using tenure_months * monthly_charges")

# 7. satisfaction_score
df.loc[~df['satisfaction_score'].isin([1, 2, 3, 4, 5]), 'satisfaction_score'] = np.nan
median_score = df['satisfaction_score'].median()
df['satisfaction_score'] = df['satisfaction_score'].fillna(median_score).round().astype(int)
print(f"Satisfaction Score: imputed missing with median ({median_score:.0f}) and cast to int64")"""))

    cells.append(nbf.v4.new_code_cell("""# 8. join_date
df['join_date'] = pd.to_datetime(df['join_date'], format='mixed', errors='coerce')
ref_date = pd.Timestamp("2024-12-31")
missing_dates = df['join_date'].isnull().sum()
if missing_dates > 0:
    imputed_dates = ref_date - pd.to_timedelta(df.loc[df['join_date'].isnull(), 'tenure_months'] * 30.4375, unit='D')
    df.loc[df['join_date'].isnull(), 'join_date'] = imputed_dates
print(f"Join Date: converted heterogeneous date strings to datetime64[ns] ({missing_dates} imputed from tenure)")"""))

    cells.append(nbf.v4.new_code_cell("""# 9. Categorical standardization
# Gender
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

# Contract Type
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

# Payment Method
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

# Internet Service & Tech Support
df['internet_service'] = df['internet_service'].apply(lambda s: "Fiber Optic" if "fiber" in str(s).lower() else ("DSL" if "dsl" in str(s).lower() else "No"))
df['tech_support'] = df['tech_support'].apply(lambda s: "Yes" if str(s).strip().lower() in ["yes", "y", "true", "1"] else ("No internet service" if "no internet" in str(s).lower() else "No"))

# City & Churn
df['city'] = df['city'].astype(str).apply(lambda x: "Unknown" if x.strip().lower() in ["nan", "null", "none", ""] else x.strip().title())
df['churn'] = df['churn'].apply(lambda s: "Yes" if str(s).strip().lower() in ["yes", "y", "true", "1"] else "No")

print("Categorical standardization complete.")
print(df[['gender', 'contract_type', 'payment_method', 'internet_service', 'tech_support', 'churn']].nunique())"""))

    # Step 6: Outlier Detection & Handling (IQR & Z-score)
    cells.append(nbf.v4.new_markdown_cell("""## Step 6: Detect & Handle Outliers Using IQR and Z-Score Methods
We evaluate numerical features (`age`, `annual_income`, `tenure_months`, `monthly_charges`, `total_charges`) for anomalies.
We employ both:
1. **IQR Method**:
   $$\\text{IQR} = Q_3 - Q_1$$
   $$\\text{Lower Bound} = Q_1 - 1.5 \\times \\text{IQR}, \\quad \\text{Upper Bound} = Q_3 + 1.5 \\times \\text{IQR}$$
2. **Z-Score Method**:
   $$Z = \\frac{x - \\mu}{\\sigma}, \\quad |Z| > 3$$

We then apply **Winsorization / IQR Capping** to constrain severe financial outliers without dropping valuable customer records.
"""))

    cells.append(nbf.v4.new_code_cell("""num_cols = ["age", "annual_income", "tenure_months", "monthly_charges", "total_charges"]

# Boxplots BEFORE outlier treatment
fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="#38BDF8", flierprops={"marker": "o", "color": "#E11D48", "markersize": 5})
    axes[i].set_title(col.replace('_', ' ').title(), fontsize=11, fontweight="bold")
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)
plt.suptitle("Outlier Inspection via Boxplots (Before Treatment)", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Compute IQR and Z-Score Statistics
outlier_metrics = []
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

    outlier_metrics.append({
        "Feature": col,
        "Mean": mean_val,
        "Std Dev": std_val,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "IQR Lower Bound": lower_bound,
        "IQR Upper Bound": upper_bound,
        "IQR Outliers (Count)": iqr_outliers,
        "Z-Score (|Z|>3) Outliers": z_outliers
    })

outlier_table = pd.DataFrame(outlier_metrics)
display(outlier_table)"""))

    cells.append(nbf.v4.new_code_cell("""# Apply Winsorization / IQR Capping to financial variables
for col in ["annual_income", "monthly_charges"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = max(0.0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    initial_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"Capped {initial_clipped} values in '{col}' to range [{lower:.2f}, {upper:.2f}].")

# Total charges re-capping
df['total_charges'] = df['total_charges'].clip(lower=0.0, upper=df['tenure_months'].max() * df['monthly_charges'].max())

# Boxplots AFTER outlier treatment
fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i], color="#34D399", flierprops={"marker": "o", "color": "#059669", "markersize": 4})
    axes[i].set_title(col.replace('_', ' ').title(), fontsize=11, fontweight="bold")
    axes[i].grid(axis="y", linestyle="--", alpha=0.6)
plt.suptitle("Outlier Inspection via Boxplots (After IQR Capping Treatment)", fontsize=14, fontweight="bold", y=1.03)
plt.tight_layout()
plt.show()"""))

    # Step 7: Categorical Encoding & Normalization
    cells.append(nbf.v4.new_markdown_cell("""## Step 7: Normalize or Encode Categorical Variables
To prepare the dataset for predictive machine learning models, we implement:
1. **Ordinal Encoding**: For ordered factors like `contract_type` (Month-to-Month: 0, One Year: 1, Two Year: 2).
2. **Binary Encoding**: For binary target `churn` (No: 0, Yes: 1).
3. **One-Hot Encoding**: For nominal categories (`gender`, `internet_service`, `tech_support`, `payment_method`) using `pd.get_dummies(..., drop_first=True)`.
4. **Min-Max Feature Scaling**: Transforming numerical features into the standard $[0, 1]$ interval.
"""))

    cells.append(nbf.v4.new_code_cell("""df_encoded = df.copy()

# 1. Ordinal Encoding
contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
df_encoded['contract_type_encoded'] = df_encoded['contract_type'].map(contract_map)

# 2. Binary Encoding
churn_map = {"No": 0, "Yes": 1}
df_encoded['churn_encoded'] = df_encoded['churn'].map(churn_map)

# 3. One-Hot Encoding
nominal_cols = ["gender", "internet_service", "tech_support", "payment_method"]
df_encoded = pd.get_dummies(df_encoded, columns=nominal_cols, drop_first=True, dtype=int)

# 4. Min-Max Normalization
for col in ["annual_income", "monthly_charges", "total_charges"]:
    c_min = df_encoded[col].min()
    c_max = df_encoded[col].max()
    df_encoded[f"{col}_normalized"] = (df_encoded[col] - c_min) / (c_max - c_min)

print("ML-Ready Encoded Dataset Dimensions:", df_encoded.shape)
df_encoded.head()"""))

    # Step 8: Validate Cleaned Dataset
    cells.append(nbf.v4.new_markdown_cell("""## Step 8: Validate the Cleaned Dataset with Summary Statistics
We run comprehensive integrity assertions and inspect final distributions to confirm the dataset is clean and analysis-ready.
"""))

    cells.append(nbf.v4.new_code_cell("""# Summary statistics of clean dataset
display(df.describe().T)"""))

    cells.append(nbf.v4.new_code_cell("""# Automated Data Quality Assertions
assert df.isnull().sum().sum() == 0, "Data validation failed: Null values remain!"
assert df.duplicated().sum() == 0, "Data validation failed: Duplicate rows remain!"
assert df.duplicated(subset=['customer_id']).sum() == 0, "Data validation failed: Duplicate IDs remain!"
assert (df['age'] < 18).sum() == 0, "Data validation failed: Minor age detected!"
assert (df['annual_income'] < 0).sum() == 0, "Data validation failed: Negative income detected!"
assert (df['monthly_charges'] <= 0).sum() == 0, "Data validation failed: Non-positive charge detected!"

print(" ALL 6 DATA INTEGRITY ASSERTIONS PASSED SUCCESSFULLY!")
print(f"Total Clean Records: {len(df):,}")
print(f"Null Values Remaining: {df.isnull().sum().sum()} (0.00%)")
print(f"Primary Key Uniqueness: {df['customer_id'].nunique():,} / {len(df):,} (100.0%)")"""))

    cells.append(nbf.v4.new_code_cell("""# Plot final distributions of cleaned numerical & ordinal features
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color="#2563EB", bins=25, edgecolor="#1E40AF")
    axes[i].set_title(f"Distribution of {col.replace('_', ' ').title()}", fontsize=11, fontweight="bold")
    axes[i].set_xlabel(col.replace('_', ' ').title(), fontsize=10)
    axes[i].set_ylabel("Frequency", fontsize=10)
    axes[i].grid(axis="y", linestyle="--", alpha=0.5)

sns.countplot(x=df['satisfaction_score'], ax=axes[5], palette="Blues_d", hue=df['satisfaction_score'], legend=False)
axes[5].set_title("Satisfaction Score Distribution (1-5)", fontsize=11, fontweight="bold")
axes[5].set_xlabel("Satisfaction Score", fontsize=10)
axes[5].set_ylabel("Count", fontsize=10)
axes[5].grid(axis="y", linestyle="--", alpha=0.5)

plt.suptitle("Post-Cleaning Distributions of Numerical & Ordinal Attributes", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()"""))

    # Step 9: Export Cleaned Dataset
    cells.append(nbf.v4.new_markdown_cell("""## Step 9: Export the Cleaned Dataset to CSV
We persist both the clean analysis-ready dataset and the ML-ready encoded dataset to disk.
"""))

    cells.append(nbf.v4.new_code_cell("""cleaned_path = '../data/cleaned/customer_churn_cleaned.csv'
encoded_path = '../data/cleaned/customer_churn_encoded.csv'

df.to_csv(cleaned_path, index=False)
df_encoded.to_csv(encoded_path, index=False)

print(f"Saved Cleaned Dataset: {cleaned_path} ({os.path.getsize(cleaned_path):,} bytes)")
print(f"Saved Encoded Dataset: {encoded_path} ({os.path.getsize(encoded_path):,} bytes)")"""))

    # Step 10: Before-and-After Comparison Table & Report
    cells.append(nbf.v4.new_markdown_cell("""## Step 10: Before-and-After Comparison Table of Data Quality Metrics
The table below encapsulates the complete transformation across all quality dimensions.
"""))

    cells.append(nbf.v4.new_code_cell("""comparison_df = pd.read_csv('../outputs/tables/before_after_metrics.csv')
display(comparison_df)"""))

    cells.append(nbf.v4.new_markdown_cell("""## Conclusion & Next Steps
The raw, messy customer churn dataset has been successfully transformed into a rigorous, production-grade format:
- **Zero Missing Values**: Imputed via statistical medians, domain formulas, and documented categories.
- **Zero Duplicates**: Exact duplicates and primary key collisions purged.
- **Standardized Schema**: Normalized to clean `snake_case` with verified data types.
- **Tamed Outliers**: Winsorized extreme billing glitches and typos using IQR clipping.
- **Dual Outputs**: Provided both human-readable analytical CSV and encoded numerical ML-ready CSV.
"""))

    nb.cells = cells

    # Save unexecuted notebook
    os.makedirs(os.path.dirname(notebook_path), exist_ok=True)
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Wrote notebook structure to: {notebook_path}")

    # Execute the notebook using NotebookClient
    print("Executing notebook cells to render all outputs, tables, and figures...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": os.path.dirname(notebook_path)}})
    client.execute()

    # Save executed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved notebook with pre-rendered outputs: {notebook_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    nb_file = os.path.join(base_dir, "notebooks", "data_cleaning_project.ipynb")
    create_and_execute_notebook(nb_file)
