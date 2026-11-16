"""
End-to-end attrition pipeline.

    1. Generate synthetic HR data (if missing)
    2. Build the five SQL views for two landmark dates
         train landmark 2024-06-30 (24 months of follow-up)
         OOT   landmark 2025-06-30 (12 months of follow-up)
    3. Cox PH on the train landmark -> hazard ratios, PH test, OOT concordance
    4. XGBoost triage (exit within 6 months) -> OOT AUC, precision of the top-50 list, SHAP
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from src.features import build_model_base, connect, to_matrix
from src.models import fit_cox, fit_triage, ph_assumption

ROOT = Path(__file__).parent
REPORTS, FIG = ROOT / "reports", ROOT / "reports" / "figures"
TRAIN_LM, OOT_LM = "2024-06-30", "2025-06-30"
ACCENT = "#E8A54B"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "data/raw/employee_details.parquet").exists():
        subprocess.run([sys.executable, str(ROOT / "data/generate_synthetic.py")], check=True)

    con = connect()
    raw = con.execute("SELECT COALESCE(exit_reason,'ACTIVE') r, COUNT(*) n FROM employee_details GROUP BY 1 ORDER BY 2 DESC").df()
    tr = build_model_base(con, TRAIN_LM)
    oot = build_model_base(con, OOT_LM)
    print(f"train landmark {TRAIN_LM}: {len(tr):,} employees, 24m attrition {tr.event.mean():.1%}, 6m {tr.exit_within_horizon.mean():.1%}")
    print(f"OOT landmark   {OOT_LM}: {len(oot):,} employees, 12m attrition {oot.event.mean():.1%}, 6m {oot.exit_within_horizon.mean():.1%}")

    # ---------------- Cox PH ----------------
    cph, d_tr = fit_cox(tr)
    _, d_oot = fit_cox(oot)  # only to get the same transformed columns
    risk_oot = cph.predict_partial_hazard(d_oot.drop(columns=["duration_months", "event"]))
    c_oot = concordance_index(d_oot.duration_months, -risk_oot, d_oot.event)
    hr = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
        columns={"exp(coef)": "hazard_ratio", "exp(coef) lower 95%": "ci_low", "exp(coef) upper 95%": "ci_high"}).round(3)
    hr = hr.sort_values("hazard_ratio", ascending=False)
    ph = ph_assumption(cph, d_tr)
    print(hr); print(f"Cox concordance: train {cph.concordance_index_:.3f} | OOT {c_oot:.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    y = np.arange(len(hr))
    ax.errorbar(hr.hazard_ratio, y, xerr=[hr.hazard_ratio - hr.ci_low, hr.ci_high - hr.hazard_ratio], fmt="o",
                color=ACCENT, ecolor="#8A8A8A", capsize=3)
    ax.axvline(1, color="grey", ls="--", lw=1); ax.set_yticks(y, hr.index); ax.set_xscale("log")
    ax.set_xlabel("hazard ratio (log scale)"); ax.set_title("What multiplies the risk of leaving (Cox PH)")
    fig.tight_layout(); fig.savefig(FIG / "cox_hazard_ratios.png"); plt.close(fig)

    # KM: underpaid high performers vs everyone else
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for flag, lab, c in [(1, "underpaid high performers", "#D9534F"), (0, "everyone else", "#5B8DEF")]:
        m = tr.underpaid_high_performer == flag
        KaplanMeierFitter().fit(tr.duration_months[m], tr.event[m], label=f"{lab} (n={m.sum()})").plot_survival_function(ax=ax, color=c)
    ax.set_xlabel("months after landmark"); ax.set_ylabel("share still employed"); ax.set_title("Retention curves")
    fig.tight_layout(); fig.savefig(FIG / "km_underpaid_high_performers.png"); plt.close(fig)

    # ---------------- XGBoost triage ----------------
    Xt = to_matrix(tr)
    Xo = to_matrix(oot).reindex(columns=Xt.columns, fill_value=0.0)
    model = fit_triage(Xt, tr.exit_within_horizon)
    p = model.predict_proba(Xo)[:, 1]
    auc = roc_auc_score(oot.exit_within_horizon, p)
    top = pd.DataFrame({"p": p, "y": oot.exit_within_horizon.to_numpy()}).sort_values("p", ascending=False)
    k = 50
    prec_k = top.head(k).y.mean()
    top_decile_capture = top.head(len(top) // 10).y.sum() / max(top.y.sum(), 1)
    print(f"Triage OOT AUC {auc:.3f} | precision@{k} {prec_k:.1%} vs base {oot.exit_within_horizon.mean():.1%} "
          f"| top-decile capture {top_decile_capture:.1%}")

    sv = shap.TreeExplainer(model).shap_values(Xo)
    shap.summary_plot(sv, Xo, show=False, max_display=12)
    plt.title("Monthly triage: SHAP drivers"); plt.tight_layout()
    plt.savefig(FIG / "shap_triage.png", bbox_inches="tight"); plt.close()
    reasons = [", ".join(Xo.columns[np.argsort(-r)[:3]][r[np.argsort(-r)[:3]] > 0]) for r in sv]

    triage = oot[["emp_id", "department", "grade"]].assign(risk_6m=p.round(4), top_reasons=reasons)
    triage.sort_values("risk_6m", ascending=False).head(200).to_csv(REPORTS / "triage_list_oot.csv", index=False)
    hr.to_csv(REPORTS / "cox_hazard_ratios.csv")

    metrics = {
        "raw_exit_reasons": dict(zip(raw.r, raw.n.astype(int))),
        "train_landmark": TRAIN_LM, "oot_landmark": OOT_LM,
        "employees_train": int(len(tr)), "employees_oot": int(len(oot)),
        "attrition_24m_train": round(float(tr.event.mean()), 4),
        "cox_concordance_train": round(float(cph.concordance_index_), 4),
        "cox_concordance_oot": round(float(c_oot), 4),
        "cox_hazard_ratios": hr.reset_index().rename(columns={"covariate": "feature"}).to_dict(orient="records"),
        "ph_test_min_p": float(ph.p.min()),
        "triage_auc_oot": round(float(auc), 4),
        "triage_precision_at_50": round(float(prec_k), 4),
        "triage_base_rate_6m": round(float(oot.exit_within_horizon.mean()), 4),
        "triage_top_decile_capture": round(float(top_decile_capture), 4),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


if __name__ == "__main__":
    main()
