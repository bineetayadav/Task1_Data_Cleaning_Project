"""
generate_raw_data.py
Synthesizes a realistic, messy real-world customer churn & demographic dataset
containing authentic real-world data flaws:
- Inconsistent and messy column headers
- Missing and null values across numerical, categorical, and date fields
- Exact duplicate rows and duplicate identifier records
- Mixed and non-standard data types (currency strings with '$' and ',', mixed date formats)
- Inconsistent categorical text representations and casing variations
- Statistical and domain-specific outliers (e.g., negative ages, extreme incomes, impossible scores)
"""

import os
import random
import numpy as np
import pandas as pd

def generate_messy_customer_dataset(output_path: str, n_rows: int = 1500, random_seed: int = 42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 1. Base entities
    first_names = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Aarav", "Priya",
        "Rohit", "Ananya", "Vikram", "Sneha", "Aditya", "Pooja", "Arjun", "Neha"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Sharma", "Verma", "Patel", "Reddy", "Nair", "Iyer", "Gupta", "Singh", "Kumar"
    ]
    cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Chennai",
        "London", "Toronto", "Sydney", "Singapore", "Berlin"
    ]
    
    data = []
    
    for i in range(1, n_rows + 1):
        cust_id = f"CUST-{1000 + i}"
        
        # Name with realistic variations (leading/trailing spaces, lowercase, extra spaces)
        fname = random.choice(first_names)
        lname = random.choice(last_names)
        name_choice = random.random()
        if name_choice < 0.15:
            full_name = f"  {fname} {lname}  "
        elif name_choice < 0.25:
            full_name = f"{fname.lower()} {lname.lower()}"
        elif name_choice < 0.30:
            full_name = f"{fname.upper()} {lname.upper()}"
        elif name_choice < 0.35:
            full_name = f"{fname}  {lname}"  # double space
        else:
            full_name = f"{fname} {lname}"
            
        # Age: Normal 18-75, with flaws: negatives (-5, -2), extreme outliers (135, 142, 220), or null
        age_r = random.random()
        if age_r < 0.06:
            age = np.nan
        elif age_r < 0.08:
            age = -random.randint(1, 10)  # negative age entry error
        elif age_r < 0.10:
            age = random.choice([135, 142, 180, 225])  # extreme outlier typo
        else:
            age = int(np.random.normal(41, 14))
            age = max(18, min(75, age))
            
        # Gender with casing and abbreviation variations
        gender_r = random.random()
        if gender_r < 0.05:
            gender = np.nan
        elif gender_r < 0.45:
            gender = random.choice(["Male", "M", "male", "MALE", "m"])
        elif gender_r < 0.85:
            gender = random.choice(["Female", "F", "female", "FEMALE", "f"])
        else:
            gender = random.choice(["Other", "Non-Binary", "other"])
            
        # Annual Income ($): Currency string formatting with '$' and ',', some negative or extreme outliers
        income_r = random.random()
        if income_r < 0.08:
            annual_income = np.nan
        elif income_r < 0.10:
            annual_income = f"-${random.randint(5000, 25000):,}"  # negative income error
        elif income_r < 0.13:
            annual_income = f"${random.randint(1500000, 3500000):,}"  # multi-million outlier
        else:
            base_income = int(np.random.gamma(shape=5.0, scale=14000)) + 20000
            if random.random() < 0.5:
                annual_income = f"${base_income:,}"
            elif random.random() < 0.8:
                annual_income = f"${base_income:,.2f}"
            else:
                annual_income = f"{base_income}"  # raw number string
                
        # Tenure Months: 0 to 72, with some negatives, floats, or nulls
        tenure_r = random.random()
        if tenure_r < 0.07:
            tenure = np.nan
        elif tenure_r < 0.09:
            tenure = -random.randint(1, 6)
        elif tenure_r < 0.12:
            tenure = round(random.uniform(1.0, 72.0), 1)
        else:
            tenure = random.randint(1, 72)
            
        # Contract Type: Inconsistent representations
        contract_r = random.random()
        if contract_r < 0.05:
            contract = np.nan
        elif contract_r < 0.50:
            contract = random.choice(["Month-to-Month", "month-to-month", "Month to Month", "m2m", "M2M"])
        elif contract_r < 0.75:
            contract = random.choice(["One Year", "one-year", "1-yr", "1 Year", "1 year"])
        else:
            contract = random.choice(["Two Year", "two-year", "2-yr", "2 Year", "2 year"])
            
        # Payment Method
        pm_r = random.random()
        if pm_r < 0.06:
            payment_method = np.nan
        elif pm_r < 0.35:
            payment_method = random.choice(["Electronic Check", "electronic check", "E-Check"])
        elif pm_r < 0.60:
            payment_method = random.choice(["Mailed Check", "mailed check", "Check"])
        elif pm_r < 0.82:
            payment_method = random.choice(["Bank Transfer (automatic)", "Bank Transfer", "bank transfer"])
        else:
            payment_method = random.choice(["Credit Card (automatic)", "Credit Card", "credit card"])
            
        # Monthly Charges ($): Float, with nulls, extreme outliers (e.g., $999.99), or zeroes
        mc_r = random.random()
        if mc_r < 0.06:
            monthly_charges = np.nan
        elif mc_r < 0.08:
            monthly_charges = 0.0
        elif mc_r < 0.11:
            monthly_charges = random.choice([499.00, 750.50, 999.99, 1250.00])  # billing glitch outlier
        else:
            monthly_charges = round(random.uniform(18.50, 118.75), 2)
            
        # Total Charges: Formatted as string with spaces/blanks, nulls, or formula values
        tc_r = random.random()
        if tc_r < 0.08:
            total_charges = " "  # empty string blank flaw common in telecom datasets
        elif tc_r < 0.12:
            total_charges = np.nan
        else:
            # Approximate tenure * monthly_charges
            t_val = tenure if (pd.notnull(tenure) and tenure > 0) else random.randint(1, 24)
            m_val = monthly_charges if (pd.notnull(monthly_charges) and monthly_charges > 0) else 50.0
            calc_val = round(t_val * m_val + random.uniform(-10, 10), 2)
            calc_val = max(0.0, calc_val)
            if random.random() < 0.3:
                total_charges = f"${calc_val:,.2f}"
            else:
                total_charges = f"{calc_val:.2f}"
                
        # Internet Service
        is_r = random.random()
        if is_r < 0.05:
            internet_service = np.nan
        elif is_r < 0.45:
            internet_service = random.choice(["Fiber Optic", "fiber optic", "Fiber", "FIBER"])
        elif is_r < 0.80:
            internet_service = random.choice(["DSL", "dsl", "Dsl"])
        else:
            internet_service = random.choice(["No", "None", "no", "NONE"])
            
        # Tech Support
        ts_r = random.random()
        if ts_r < 0.07:
            tech_support = np.nan
        elif ts_r < 0.45:
            tech_support = random.choice(["Yes", "yes", "Y", "TRUE", "1"])
        elif ts_r < 0.80:
            tech_support = random.choice(["No", "no", "N", "FALSE", "0"])
        else:
            tech_support = random.choice(["No internet service", "no internet", "None"])
            
        # Satisfaction Score: 1-5 scale, with nulls, 0, or out of bound values like 9 or 10
        ss_r = random.random()
        if ss_r < 0.08:
            satisfaction = np.nan
        elif ss_r < 0.10:
            satisfaction = random.choice([0, 9, 10, -1])  # invalid scale entry
        else:
            satisfaction = random.choice([1, 2, 3, 4, 5])
            
        # Join Date: Inconsistent date formats
        year = random.choice([2020, 2021, 2022, 2023, 2024])
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        date_format_choice = random.random()
        if date_format_choice < 0.05:
            join_date = np.nan
        elif date_format_choice < 0.08:
            join_date = random.choice(["invalid_date", "unknown", "N/A"])
        elif date_format_choice < 0.40:
            join_date = f"{year}-{month:02d}-{day:02d}"  # YYYY-MM-DD
        elif date_format_choice < 0.65:
            join_date = f"{day:02d}/{month:02d}/{year}"  # DD/MM/YYYY
        elif date_format_choice < 0.85:
            join_date = f"{month:02d}-{day:02d}-{year}"  # MM-DD-YYYY
        else:
            join_date = f"{year}.{month:02d}.{day:02d}"  # YYYY.MM.DD
            
        # City
        city_r = random.random()
        if city_r < 0.06:
            city = np.nan
        else:
            c = random.choice(cities)
            if random.random() < 0.15:
                city = c.lower()
            elif random.random() < 0.10:
                city = f"  {c}  "
            else:
                city = c
                
        # Churn (Target Variable)
        churn_r = random.random()
        if churn_r < 0.04:
            churn = np.nan  # missing target
        elif churn_r < 0.32:
            churn = random.choice(["Yes", "yes", "Y", "True", "1"])
        else:
            churn = random.choice(["No", "no", "N", "False", "0"])
            
        # Intentionally inject missing Customer_ID on 1.5% of rows
        if random.random() < 0.015:
            cust_id = np.nan
            
        row = {
            " Customer_ID ": cust_id,  # messy spaces in column name
            "Full_Name": full_name,
            "Age": age,
            "Gender": gender,
            "Annual Income ($)": annual_income,  # special chars in column name
            "Tenure Months": tenure,  # space in column name
            "Contract-Type": contract,  # hyphen in column name
            "Payment_Method": payment_method,
            "Monthly Charges ($)": monthly_charges,  # special chars and spaces
            "Total Charges": total_charges,
            "Internet Service": internet_service,
            "Tech Support": tech_support,
            "Satisfaction Score (1-5)": satisfaction,
            "Join Date": join_date,
            "City": city,
            "Churn": churn
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # 2. Inject exact duplicate rows (approx 35 duplicate rows)
    dup_indices = np.random.choice(df.index, size=35, replace=False)
    duplicate_rows = df.loc[dup_indices].copy()
    
    # 3. Inject partial duplicate records with identical Customer_ID
    partial_dup_indices = np.random.choice(df.index, size=15, replace=False)
    partial_duplicates = df.loc[partial_dup_indices].copy()
    # vary monthly charges slightly
    partial_duplicates["Monthly Charges ($)"] = partial_duplicates["Monthly Charges ($)"].apply(
        lambda x: x + 1.5 if pd.notnull(x) and isinstance(x, (int, float)) else x
    )
    
    # Concatenate all rows
    df_raw = pd.concat([df, duplicate_rows, partial_duplicates], ignore_index=True)
    
    # Shuffle dataframe
    df_raw = df_raw.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_raw.to_csv(output_path, index=False)
    print(f"Successfully generated raw messy dataset: {output_path}")
    print(f"Total Rows: {len(df_raw)}, Total Columns: {len(df_raw.columns)}")
    return df_raw

if __name__ == "__main__":
    output_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "customer_churn_raw.csv"))
    generate_messy_customer_dataset(output_csv)
