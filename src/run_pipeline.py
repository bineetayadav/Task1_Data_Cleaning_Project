"""
run_pipeline.py
Master automation script for Week 2 Task 1: Data Cleaning Project.
Executes the complete pipeline in sequence:
1. Generates realistic raw messy dataset (data/raw/customer_churn_raw.csv)
2. Executes full cleaning pipeline & generates diagnostic figures + tables
3. Compiles and executes top-to-bottom Jupyter Notebook (notebooks/data_cleaning_project.ipynb)
4. Generates executive PDF report (reports/data_cleaning_report.pdf)
"""

import os
import sys
import subprocess
import time

def run_step(step_name: str, script_name: str, base_dir: str):
    print("\n" + "=" * 80)
    print(f"RUNNING: {step_name} ({script_name})")
    print("=" * 80)
    start_time = time.time()
    script_path = os.path.join(base_dir, "src", script_name)
    result = subprocess.run([sys.executable, script_path], cwd=base_dir, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:\n", result.stderr)
        
    if result.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed with return code {result.returncode}!")
        
    elapsed = time.time() - start_time
    print(f" COMPLETED: {step_name} in {elapsed:.2f} seconds.")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("=" * 80)
    print("STARTING END-TO-END DATA CLEANING PIPELINE")
    print(f"Base Directory: {base_dir}")
    print("=" * 80)
    total_start = time.time()

    run_step("1. Raw Data Synthesis", "generate_raw_data.py", base_dir)
    run_step("2. Data Cleaning & Diagnostics Pipeline", "clean_pipeline.py", base_dir)
    run_step("3. Jupyter Notebook Construction & Execution", "build_notebook.py", base_dir)
    run_step("4. Executive PDF Report Compilation", "generate_pdf_report.py", base_dir)

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print(f" ALL TASKS COMPLETED SUCCESSFULLY in {total_time:.2f}s!")
    print("=" * 80)
    print("Deliverables Generated:")
    print(f" - Raw Dataset: {os.path.join(base_dir, 'data', 'raw', 'customer_churn_raw.csv')}")
    print(f" - Cleaned Dataset: {os.path.join(base_dir, 'data', 'cleaned', 'customer_churn_cleaned.csv')}")
    print(f" - Encoded ML Dataset: {os.path.join(base_dir, 'data', 'cleaned', 'customer_churn_encoded.csv')}")
    print(f" - Jupyter Notebook: {os.path.join(base_dir, 'notebooks', 'data_cleaning_project.ipynb')}")
    print(f" - Analytical Report (MD): {os.path.join(base_dir, 'reports', 'data_cleaning_report.md')}")
    print(f" - Executive PDF Report: {os.path.join(base_dir, 'reports', 'data_cleaning_report.pdf')}")
    print(f" - Comparison Metrics: {os.path.join(base_dir, 'outputs', 'tables', 'before_after_metrics.csv')}")

if __name__ == "__main__":
    main()
