import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

DATA_FOLDER = "sales_data"
QA_CSV = "qa_output.csv"

# --- Helper: find a column from candidates ---
def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# --- Load Excel files safely ---
def load_sales_data(folder=DATA_FOLDER):
    files = [f for f in glob.glob(os.path.join(folder, "*.xlsx")) if "~$" not in f]
    if not files:
        raise FileNotFoundError(f"No Excel files found in '{folder}'")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_excel(f))
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
    if not frames:
        raise ValueError("No readable Excel files found.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    return df

# --- Load Q&A CSV (for chatbot) ---
def load_qa(csv_path=QA_CSV):
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Chat answers will be unavailable.")
        return None
    qa = pd.read_csv(csv_path)
    qa = qa.loc[:, ~qa.columns.str.contains('^Unnamed', case=False, regex=True)]
    if 'Question' in qa.columns and 'Answer' in qa.columns:
        qa["Question"] = qa["Question"].astype(str).str.strip()
        qa["Answer"] = qa["Answer"].astype(str).str.strip()
        return qa
    print("Warning: qa_output.csv does not have Question/Answer columns.")
    return None

# --- Safe anomaly detection logic (returns anomalies dataframe) ---
def detect_anomalies(df):
    # Candidate column names
    date_candidates = ["date", "order_date", "sales_date", "cy_date"]
    market_candidates = ["markets_name", "market", "market_name", "region", "zone"]
    revenue_candidates = ["norm_sales_amount", "sales_amount", "amount", "revenue"]

    date_col = find_column(df, date_candidates)
    market_col = find_column(df, market_candidates)
    revenue_col = find_column(df, revenue_candidates)

    if not date_col or not market_col or not revenue_col:
        missing = []
        if not date_col: missing.append("date")
        if not market_col: missing.append("market")
        if not revenue_col: missing.append("revenue")
        raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Work on a copy
    work = df[[date_col, market_col, revenue_col]].copy()
    work = work.rename(columns={date_col: "date", market_col: "market", revenue_col: "revenue"})

    # Coerce types safely:
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["market"] = work["market"].astype(str).fillna("Unknown Market").replace("nan", "Unknown Market")
    work["revenue"] = pd.to_numeric(work["revenue"], errors="coerce")

    # If date is missing, try to infer with forward fill; otherwise drop those rows (but don't drop everything)
    if work["date"].isna().all():
        # no valid dates at all -> set to today's date so grouping still works
        work["date"] = pd.Timestamp(datetime.now().date())
    else:
        # fill missing dates from nearest valid date (forward then backward)
        work["date"] = work["date"].fillna(method="ffill").fillna(method="bfill")

    # Replace NaN revenues with 0 (we avoid dropping rows to prevent empty datasets)
    work["revenue"] = work["revenue"].fillna(0.0)

    # Group by date & market
    sales = work.groupby(["date", "market"], as_index=False)["revenue"].sum()

    if sales.empty:
        raise ValueError("No grouped sales data available after processing.")

    # If dataset small, use z-score fallback
    if len(sales) < 5:
        # compute z-score on revenue
        mu = sales["revenue"].mean()
        sigma = sales["revenue"].std(ddof=0) if sales["revenue"].std(ddof=0) != 0 else 1.0
        sales["zscore"] = (sales["revenue"] - mu) / sigma
        # mark anomalies as abs(z) > 2.5 (tuneable)
        sales["anomaly"] = sales["zscore"].abs() > 2.5
        anomalies = sales[sales["anomaly"]].copy()
        anomalies = anomalies.sort_values(by="revenue", ascending=False)
        anomalies = anomalies.reset_index(drop=True)
        return sales, anomalies, "zscore"
    else:
        # IsolationForest path (requires >= 2 samples; more reliable with >20)
        try:
            model = IsolationForest(contamination=0.05, random_state=42)
            sales["anomaly_score"] = model.fit_predict(sales[["revenue"]])
            # model returns -1 for anomaly
            anomalies = sales[sales["anomaly_score"] == -1].copy()
            anomalies = anomalies.sort_values(by="revenue", ascending=False).reset_index(drop=True)
            return sales, anomalies, "isolation_forest"
        except Exception as e:
            # fallback to z-score if any error
            mu = sales["revenue"].mean()
            sigma = sales["revenue"].std(ddof=0) if sales["revenue"].std(ddof=0) != 0 else 1.0
            sales["zscore"] = (sales["revenue"] - mu) / sigma
            sales["anomaly"] = sales["zscore"].abs() > 2.5
            anomalies = sales[sales["anomaly"]].copy()
            anomalies = anomalies.sort_values(by="revenue", ascending=False).reset_index(drop=True)
            return sales, anomalies, f"fallback_zscore_due_to_{e}"

# --- Save anomalies and print summary ---
def save_and_report(anomalies, method):
    if anomalies is None or anomalies.empty:
        print("\n✅ No anomalies found.")
        return
    csv_out = "anomaly_output.csv"
    xlsx_out = "anomaly_output.xlsx"
    anomalies.to_csv(csv_out, index=False)
    anomalies.to_excel(xlsx_out, index=False)
    print(f"\n🚨 {len(anomalies)} anomalies detected (method: {method})")
    print(anomalies.head(20).to_string(index=False))
    print(f"\nSaved: {csv_out} and {xlsx_out}")

# --- Chatbot minimal QA lookup ---
def ask_ai(qa_df, question):
    if qa_df is None:
        return "QA not available."
    q = str(question).strip().lower()
    match = qa_df.loc[qa_df["Question"].str.lower().str.strip() == q, "Answer"]
    if not match.empty:
        return match.values[0]
    return "Sorry, I don't have an answer for that question."

# --- Main interactive loop ---
def main():
    try:
        print("Loading data...")
        sales_df = load_sales_data()
        qa = None
        try:
            qa = load_qa()
        except Exception:
            qa = None
        print("Columns found:", list(sales_df.columns))
    except Exception as e:
        print("Fatal error while loading data:", e)
        return

    print("\nType 'anomaly' to run anomaly detection, 'stats' for summary, 'exit' to quit.")
    while True:
        cmd = input(">> ").strip()
        if not cmd:
            continue
        if cmd.lower() == "exit":
            print("Goodbye.")
            break
        if cmd.lower() == "stats":
            # quick overview
            try:
                print("\nQuick stats:")
                print("Rows loaded:", len(sales_df))
                # try to find revenue col name
                rev = find_column(sales_df, ["norm_sales_amount", "sales_amount", "amount", "revenue"])
                print("Using revenue-like column:", rev)
                if rev:
                    vals = pd.to_numeric(sales_df[rev], errors="coerce")
                    print("Total revenue (sum of numeric):", vals.sum(skipna=True))
                    print("Mean (numeric):", vals.mean(skipna=True))
            except Exception as e:
                print("Stats error:", e)
            continue
        if cmd.lower() == "anomaly":
            try:
                sales_grouped, anomalies, method = detect_anomalies(sales_df)
                save_and_report(anomalies, method)
            except Exception as e:
                print("Anomaly detection failed:", e)
            continue
        # Otherwise treat as QA question
        if qa is not None:
            print("AI:", ask_ai(qa, cmd))
        else:
            print("QA not loaded. Type 'anomaly' or 'exit'.")

if __name__ == "__main__":
    main()

