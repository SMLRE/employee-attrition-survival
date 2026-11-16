"""Build the five views in DuckDB for a landmark date and return the model base."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, SQL = ROOT / "data" / "raw", ROOT / "sql"
END_DATE = "2026-06-30"
VIEWS = ["v1_employee_clean", "v2_survival_spells", "v3_compensation", "v4_mobility", "v5_performance"]

FEATURES = ["tenure_months", "age_years", "grade", "pay_vs_grade_median", "months_since_raise",
            "months_since_promotion", "pay_growth_24m", "mgr_changes_12m", "transfers_24m",
            "months_since_mgr_change", "last_rating", "rating_change", "underpaid_high_performer"]


def connect():
    con = duckdb.connect()
    for t in ["employee_details", "pay_history", "appraisals", "assignments"]:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{RAW / (t + '.parquet')}')")
    return con


def build_model_base(con, landmark: str, end_date: str = END_DATE, horizon: int = 6) -> pd.DataFrame:
    p = dict(landmark=landmark, end_date=end_date, horizon=horizon)
    for v in VIEWS:
        con.execute((SQL / "views" / f"{v}.sql").read_text().format(**p))
    df = con.execute((SQL / "model_base.sql").read_text().format(**p)).df()
    df["landmark"] = landmark
    return df


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].astype(float).fillna({"rating_change": 0, "last_rating": 3, "pay_growth_24m": 0})
    return pd.concat([X, pd.get_dummies(df.department, prefix="dept", dtype=float)], axis=1)
