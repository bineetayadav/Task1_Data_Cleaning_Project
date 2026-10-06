# Task 1: End-to-End Data Cleaning Project

**Repository**: [Task1_Data_Cleaning_Project](https://github.com/bineetayadav/Task1_Data_Cleaning_Project)  
**Track**: Data Science & Machine Learning Engineering  
**Module**: Advanced Data Preparation, Quality Assurance & Feature Preprocessing  
**Author**: Bineeta Yadav (Data Science Intern)  
**Date**: October 2026  
**Language & Tools**: Python 3.12, Pandas 2.3, NumPy 2.2, Seaborn 0.13, Matplotlib 3.11, Jupyter, ReportLab  

---

## 📌 Project Overview & Objective

This repository contains the complete, production-grade submission for **Task 1: Data Cleaning Project**.

The core objective is to ingest a raw, messy real-world enterprise dataset (`customer_churn_raw.csv`), identify its structural flaws and statistical distortions, and transform it into a pristine, analysis-ready and machine-learning-ready format using Python's **Pandas** library.

### Key Highlights:
- **1,550 Raw Records** ingested with 16 attributes spanning customer demographics, subscription contract terms, service configurations, payment telemetry, and churn outcomes.
- **1,522 Missing Values Resolved** (6.13% missing rate in raw data) with 0.00% residual nulls achieved through statistical medians, domain formula reconstruction, and explicit category classification.
- **Deduplication**: 35 exact duplicate rows and 14 primary key (`customer_id`) collisions purged.
- **Syntactic Normalization**: All column names standardized to uniform `snake_case`.
- **Data Type Corrections**: Corrupted string currencies (`"$85,000"`), whitespace blanks (`" "`), and mixed date strings (`YYYY-MM-DD`, `DD/MM/YYYY`, `MM-DD-YYYY`) parsed and cast to `float64`, `int64`, and `datetime64[ns]`.
- **Outlier Treatment**: Dual evaluation via **Interquartile Range (IQR)** and **Z-Score ($|Z| > 3$)**, coupled with **Winsorization (IQR Capping)** on financial attributes.
- **Feature Preprocessing**: Ordinal encoding on contract tiers, binary mapping on churn, one-hot encoding on nominal factors, and Min-Max feature scaling.

---

## 📊 Before-and-After Comparison of Data Quality Metrics

| Data Quality Dimension | Raw Messy Dataset (Before) | Cleaned Dataset (After) | Transformation Impact & Decision Rationale |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 1,550 | 1,413 | 137 records removed (35 duplicates, 14 PK collisions, 26 missing IDs, 62 missing targets). |
| **Total Columns** | 16 | 16 (Clean) / 28 (Encoded ML) | Normalized to `snake_case`; expanded with one-hot dummies for ML. |
| **Total Missing Values** | 1,522 cells (6.13%) | **0 cells (0.00%)** | 100% resolved via statistical medians, formula interpolation, and categorization. |
| **Exact Duplicate Rows** | 35 | **0** | Eliminated repeated ingestion artifacts and transaction inflation. |
| **Duplicate Customer IDs** | 74 | **0** | Enforced 1-to-1 customer entity resolution (`customer_id`). |
| **Column Naming Standard** | Inconsistent (`' Customer_ID '`, `'Annual Income ($)'`)| Clean canonical `snake_case` | Enables dot-notation programmatic attribute access (`df.annual_income`). |
| **Annual Income Data Type** | `object` (strings with `$` and `,`) | `float64` (numeric) | Enabled statistical modeling, variance analysis, and aggregation. |
| **Total Charges Data Type** | `object` (blanks `' '`, corrupted text) | `float64` (numeric) | Reconciled using domain formula: $\text{tenure\_months} \times \text{monthly\_charges}$. |
| **Join Date Data Type** | `object` (heterogeneous date formats) | `datetime64[ns]` | Enables chronological sorting, tenure auditing, and cohort analysis. |
| **Negative Age / Incomes** | Present (entry typos like `-5`) | **0 (purged / imputed)** | Enforced positive human age and income domain constraints. |
| **Extreme Outliers** | $3.5M incomes, $999/mo billing spikes | Winsorized / IQR Capped | Controlled leverage points while maintaining complete customer profiles. |
| **Categorical Consistency** | Fragmented (`'m'`, `'Male'`, `'MALE'`, `'1'`) | Uniform taxonomy (`'Male'`, `'Female'`) | Eliminated sparse, splintered factor levels. |

---

## 🛠️ Step-by-Step Implementation Workflow

The pipeline strictly fulfills all 10 required implementation steps:

```
[1. Load & Initial Explore] ──> [2. Visualize Missingness] ──> [3. Strategy Formulation]
               │                                                              │
               ▼                                                              ▼
[4. Duplicate Purge]        ──> [5. snake_case & Cast Types] ──> [6. IQR & Z-Score Outliers]
               │                                                              │
               ▼                                                              ▼
[7. Categorical Encoding]   ──> [8. Summary Validation]     ──> [9. Export Datasets & Report]
```

### 1. Initial Exploration (`.info()`, `.describe()`, `.head()`)
- Loaded raw dataset into Pandas DataFrame.
- Inspected shape (1,550 x 16), verified object-heavy column schema, and audited preliminary distribution metrics.

### 2. Missing Data Diagnostics
- Rendered a high-resolution Seaborn **Missing Data Heatmap** (`outputs/figures/01_missing_data_heatmap.png`) highlighting null distribution patterns.
- Constructed a **Missing Values Percentage Bar Chart** (`outputs/figures/02_missing_data_bar.png`) ranking features by missingness severity.

### 3. Systematic Missing Value Imputation
- **Primary Keys (`customer_id`) & Targets (`churn`)**: Dropped records with missing IDs or unlabelled targets to avoid synthetic bias.
- **Continuous Features (`annual_income`, `age`, `tenure_months`, `monthly_charges`)**: Imputed missing entries using the **median** to prevent skewness from extreme outliers.
- **Formula Reconciliation (`total_charges`)**: Reconstructed missing values using domain logic: $\text{tenure\_months} \times \text{monthly\_charges}$.
- **Temporal Backfill (`join_date`)**: Backfilled missing dates using reference cutoff minus customer tenure.
- **Categoricals (`payment_method`, `city`, `gender`, `contract_type`, etc.)**: Standardized and imputed with `'Unknown'`, `'Unspecified'`, or mode.

### 4. Duplicate Detection & Removal
- Identified and purged **35 exact duplicate rows** via `df.drop_duplicates()`.
- Screened for primary key duplicates using `df.duplicated(subset=['customer_id'])` and removed 14 secondary conflicting records.

### 5. Column Standardization & Type Corrections
- Replaced spaces, hyphens, and parentheses with single underscores using regular expressions, establishing uniform `snake_case`.
- Stripped currency symbols (`$`, `,`) and negative signs from financial columns; converted to `float64`.
- Parsed heterogeneous date formats (`YYYY-MM-DD`, `DD/MM/YYYY`, `MM-DD-YYYY`) to native `datetime64[ns]`.

### 6. Outlier Detection & IQR Winsorization
- Constructed **Pre-Treatment Boxplots** (`outputs/figures/03_outliers_before_boxplots.png`) revealing extreme leverage points.
- Computed statistical boundaries:
  - **IQR Method**: $Q_1, Q_3, \text{IQR} = Q_3 - Q_1, \text{Bounds} = [Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$.
  - **Z-Score Method**: $Z = \frac{x - \mu}{\sigma}$, identifying $|Z| > 3.0$.
- Applied **Winsorization (IQR Capping)** to constrain financial variables (`annual_income`, `monthly_charges`) to theoretical 99th percentile boundaries without dropping valuable customer profiles.
- Validated post-treatment distributions with **Post-Treatment Boxplots** (`outputs/figures/04_outliers_after_boxplots.png`).

### 7. Categorical Encoding & Normalization
- **Ordinal Encoding**: `contract_type` mapped to $\{0, 1, 2\}$ respecting commitment duration.
- **Binary Encoding**: `churn` mapped to $\{0, 1\}$.
- **One-Hot Encoding**: Applied to nominal features (`gender`, `internet_service`, `tech_support`, `payment_method`) with `drop_first=True`.
- **Min-Max Scaling**: Scaled continuous attributes to the interval $[0, 1]$ for gradient-based ML algorithms.

### 8. Validation with Summary Statistics
- Executed automated assertion suite:
  ```python
  assert df.isnull().sum().sum() == 0
  assert df.duplicated().sum() == 0
  assert df.duplicated(subset=['customer_id']).sum() == 0
  assert (df['age'] < 18).sum() == 0
  assert (df['annual_income'] < 0).sum() == 0
  assert (df['monthly_charges'] <= 0).sum() == 0
  ```
- Generated multi-panel distribution plots (`outputs/figures/05_numerical_distributions.png`).

### 9. Export Cleaned Datasets
- Clean Analysis-Ready CSV: `data/cleaned/customer_churn_cleaned.csv` (1,413 rows, 16 standardized columns).
- ML-Ready Encoded CSV: `data/cleaned/customer_churn_encoded.csv` (1,413 rows, 28 encoded/normalized features).

### 10. Comprehensive Reporting
- Complete analytical markdown report: `reports/data_cleaning_report.md`
- Multi-page executive PDF report with embedded charts and tables: `reports/data_cleaning_report.pdf`
- Before-and-after metrics table: `outputs/tables/before_after_metrics.csv`

---

## 📁 Repository Structure

```
2nd_Week_Task1/
├── data/
│   ├── raw/
│   │   └── customer_churn_raw.csv         <- Raw, messy real-world dataset (1,550 rows)
│   └── cleaned/
│       ├── customer_churn_cleaned.csv     <- Cleaned, analysis-ready dataset (1,413 rows)
│       └── customer_churn_encoded.csv     <- ML-ready dataset with encodings & scalings
├── notebooks/
│   └── data_cleaning_project.ipynb        <- Executed Jupyter Notebook with all outputs & plots
├── outputs/
│   ├── figures/
│   │   ├── 01_missing_data_heatmap.png    <- Missing data heatmap before cleaning
│   │   ├── 02_missing_data_bar.png        <- Missing percentage per column
│   │   ├── 03_outliers_before_boxplots.png<- Multi-panel boxplots before capping
│   │   ├── 04_outliers_after_boxplots.png <- Multi-panel boxplots after IQR capping
│   │   └── 05_numerical_distributions.png <- Post-cleaning feature distribution histograms
│   └── tables/
│       └── before_after_metrics.csv       <- Quality metrics comparison table
├── reports/
│   ├── data_cleaning_report.md            <- Comprehensive Markdown report with full rationale
│   └── data_cleaning_report.pdf           <- Executive PDF report with styled tables & embedded charts
├── src/
│   ├── generate_raw_data.py               <- Synthesizes realistic messy raw dataset
│   ├── clean_pipeline.py                  <- End-to-end cleaning pipeline script
│   ├── build_notebook.py                  <- Builds and executes the Jupyter Notebook
│   ├── generate_pdf_report.py             <- Compiles the executive PDF report
│   └── run_pipeline.py                    <- Master runner executing all steps sequentially
├── requirements.txt                       <- Python dependencies
└── README.md                              <- Project overview and documentation
```

---

## ⚡ Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete End-to-End Pipeline
Execute the master runner to re-synthesize data, run the cleaning pipeline, execute the notebook, and compile the PDF report in a single command:
```bash
python src/run_pipeline.py
```

### 3. Or Run Individual Pipeline Modules
```bash
# Step 1: Synthesize raw messy dataset
python src/generate_raw_data.py

# Step 2: Run cleaning pipeline & generate figures
python src/clean_pipeline.py

# Step 3: Compile and execute Jupyter Notebook
python src/build_notebook.py

# Step 4: Compile executive PDF report
python src/generate_pdf_report.py
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/data_cleaning_project.ipynb
```

---

## 🎯 Deliverables Checklist & Evaluation Alignment

- [x] **Jupyter Notebook (`.ipynb`)**: Available at `notebooks/data_cleaning_project.ipynb` with all 10 steps, markdown documentation, code cells, and pre-rendered outputs.
- [x] **Original Raw Dataset**: Located at `data/raw/customer_churn_raw.csv` (1,550 rows).
- [x] **Cleaned Output CSV Files**: Located at `data/cleaned/customer_churn_cleaned.csv` and `data/cleaned/customer_churn_encoded.csv`.
- [x] **Data Cleaning Report**: Available in Markdown (`reports/data_cleaning_report.md`) and Executive PDF (`reports/data_cleaning_report.pdf`).
- [x] **Before-and-After Comparison Table**: Saved in `outputs/tables/before_after_metrics.csv` and rendered across all reports and the notebook.
