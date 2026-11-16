"""
Two complementary models.

* Cox proportional hazards (primary): *how fast* does risk accumulate, and which
  factors multiply the hazard? Its hazard ratios are what HR policy discussions use.
* XGBoost triage (secondary): a monthly ranked list of who is most likely to
  leave in the next 6 months, with SHAP reasons, for HR business partners.
"""
from __future__ import annotations

import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

COX_FEATURES = ["tenure_months", "age_years", "grade", "pay_vs_grade_median", "months_since_raise",
                "mgr_changes_12m", "last_rating", "underpaid_high_performer"]


def fit_cox(df: pd.DataFrame, features=COX_FEATURES, penalizer: float = 0.01):
    d = df[features + ["duration_months", "event"]].copy()
    d["tenure_years"] = d.pop("tenure_months") / 12
    d["months_since_raise"] = d["months_since_raise"].clip(upper=36)
    d["low_rating"] = (d.pop("last_rating").fillna(3) <= 2).astype(int)   # new joiners: no rating yet
    d["recent_mgr_change"] = (d.pop("mgr_changes_12m") > 0).astype(int)
    d["pay_vs_grade_median"] = d["pay_vs_grade_median"] * 10  # hazard ratio per +10% of grade median
    cph = CoxPHFitter(penalizer=penalizer).fit(d, "duration_months", "event")
    return cph, d


def ph_assumption(cph, d) -> pd.DataFrame:
    res = proportional_hazard_test(cph, d, time_transform="rank")
    return res.summary[["test_statistic", "p"]].round(4)


def fit_triage(X: pd.DataFrame, y: pd.Series):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    m = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.04, subsample=0.85, colsample_bytree=0.8,
                          min_child_weight=8, reg_lambda=2.0, scale_pos_weight=spw, random_state=11, eval_metric="auc", n_jobs=1)
    return m.fit(X, y)
