"""
Synthetic HR data for the employee attrition model.

Simulates a workforce month by month from Jan 2019 to Jun 2026 (90 months):
hiring, annual appraisals, raises and promotions, transfers / manager changes,
and exits. Exit reasons mirror a real HR system, including the awkward ones:

    RESIGNATION, WITHDRAWAL_OF_SERVICE, TERMINATION   -> attrition (the target)
    RETIREMENT                                        -> deterministic (age 60), not modelled
    DEATH                                             -> excluded
    DUPLICATE_RECORD                                  -> data-quality artefact, excluded

Monthly attrition hazard rises with: time since last raise, pay below the grade
median, very low or very high (and under-paid) ratings, recent manager changes,
early tenure. It falls with promotion and seniority.

Tables written to data/raw/:
    employee_details.parquet   one row per employee record (includes duplicates)
    pay_history.parquet        hire / raise / promotion events with pay & grade
    appraisals.parquet         yearly rating 1-5
    assignments.parquet        department / manager changes
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2019-01-01")
N_MONTHS = 90
DEPTS = ["RETAIL_BRANCHES", "OPERATIONS", "TECHNOLOGY", "RISK", "FINANCE", "CORPORATE_BANKING", "HR", "TREASURY"]
GRADE_PAY = {1: 250_000, 2: 380_000, 3: 560_000, 4: 820_000, 5: 1_200_000, 6: 1_800_000}


def month_date(m):
    return START + pd.DateOffset(months=int(m))


def generate(n_initial: int = 3500, seed: int = 5):
    rng = np.random.default_rng(seed)
    emps, pay, appr, assign = [], [], [], []
    nxt = 0

    def hire(m, initial=False):
        nonlocal nxt
        eid = f"E{50000 + nxt}"; nxt += 1
        grade = int(rng.choice([1, 2, 3, 4, 5, 6], p=[0.3, 0.28, 0.2, 0.12, 0.07, 0.03]))
        age_at_hire = float(np.clip(rng.normal(27 + 4.5 * grade, 4), 21, 58))
        tenure0 = float(rng.gamma(2, 30)) if initial else 0.0
        hire_m = m - tenure0
        pay_level = GRADE_PAY[grade] * rng.lognormal(0, 0.12)
        e = dict(emp_id=eid, hire_m=hire_m, birth_m=hire_m - age_at_hire * 12, grade=grade, pay=pay_level,
                 dept=rng.choice(DEPTS), mgr=f"M{rng.integers(1, 400)}", last_raise_m=m - rng.integers(0, 14),
                 last_mgr_change_m=m - rng.integers(0, 36), rating=int(rng.choice([1, 2, 3, 4, 5], p=[.05, .15, .5, .22, .08])),
                 drive=rng.normal(0, 1), exit_m=None, reason=None)
        pay.append((eid, month_date(max(hire_m, -120)), round(pay_level, -2), grade, "HIRE"))
        assign.append((eid, month_date(max(hire_m, -120)), e["dept"], e["mgr"]))
        if initial and e["last_mgr_change_m"] > hire_m:  # record the pre-2019 manager change too
            assign.append((eid, month_date(e["last_mgr_change_m"]), e["dept"], e["mgr"]))
        return e

    active = [hire(0, initial=True) for _ in range(n_initial)]
    all_e = list(active)

    for m in range(N_MONTHS):
        # annual cycle in January: appraisals, raises, promotions
        if m % 12 == 0:
            for e in active:
                r = int(np.clip(round(rng.normal(e["rating"] * 0.6 + 1.2 + 0.3 * e["drive"], 0.8)), 1, 5))
                e["rating"] = r
                appr.append((e["emp_id"], 2019 + m // 12, r))
                if rng.random() < 0.55 + 0.08 * (r - 3):
                    e["pay"] *= 1 + rng.uniform(0.03, 0.12)
                    e["last_raise_m"] = m
                    pay.append((e["emp_id"], month_date(m), round(e["pay"], -2), e["grade"], "RAISE"))
                if r >= 4 and e["grade"] < 6 and rng.random() < 0.25:
                    e["grade"] += 1
                    e["pay"] = max(e["pay"] * 1.18, GRADE_PAY[e["grade"]] * 0.95)
                    e["last_raise_m"] = m
                    pay.append((e["emp_id"], month_date(m), round(e["pay"], -2), e["grade"], "PROMOTION"))
        still = []
        for e in active:
            # manager churn / transfers
            if rng.random() < 0.02:
                e["mgr"] = f"M{rng.integers(1, 400)}"
                if rng.random() < 0.3:
                    e["dept"] = rng.choice(DEPTS)
                e["last_mgr_change_m"] = m
                assign.append((e["emp_id"], month_date(m), e["dept"], e["mgr"]))

            age = (m - e["birth_m"]) / 12
            if age >= 60:
                e["exit_m"], e["reason"] = m, "RETIREMENT"; continue
            if rng.random() < 0.00015:
                e["exit_m"], e["reason"] = m, "DEATH"; continue

            tenure = m - e["hire_m"]
            pay_gap = np.log(e["pay"] / GRADE_PAY[e["grade"]])
            lin = (-5.6
                   + 0.06 * min(m - e["last_raise_m"], 36)
                   - 3.5 * pay_gap
                   + 0.9 * (e["rating"] >= 4) * (pay_gap < 0)
                   + 1.0 * (e["rating"] <= 2)
                   + 0.9 * (m - e["last_mgr_change_m"] <= 6)
                   + 0.6 * (tenure < 18)
                   - 0.15 * e["grade"]
                   + 0.35 * (e["dept"] in ("TECHNOLOGY", "TREASURY"))
                   + 0.25 * e["drive"])
            if rng.random() < 1 / (1 + np.exp(-lin)):
                e["exit_m"] = m
                e["reason"] = ("TERMINATION" if e["rating"] <= 2 and rng.random() < 0.6
                               else rng.choice(["RESIGNATION", "WITHDRAWAL_OF_SERVICE"], p=[0.8, 0.2]))
                continue
            still.append(e)
        # replacement hiring keeps headcount roughly stable
        new = [hire(m) for _ in range(rng.poisson(max(0, (n_initial - len(still)) * 0.35)))]
        all_e.extend(new)
        active = still + new

    rows = []
    for e in all_e:
        rows.append(dict(
            emp_id=e["emp_id"], hire_date=month_date(e["hire_m"]).normalize(), birth_date=month_date(e["birth_m"]).normalize(),
            department=e["dept"], current_grade=e["grade"],
            exit_date=month_date(e["exit_m"]) + pd.DateOffset(days=int(rng.integers(0, 27))) if e["exit_m"] is not None else pd.NaT,
            exit_reason=e["reason"]))
    emp = pd.DataFrame(rows)
    # duplicate records: ~1.5% of employees appear twice; the copy is flagged on exit
    dup = emp.sample(frac=0.015, random_state=seed).copy()
    dup["exit_reason"], dup["exit_date"] = "DUPLICATE_RECORD", dup.hire_date + pd.DateOffset(days=1)
    emp = pd.concat([emp, dup], ignore_index=True)

    pay_df = pd.DataFrame(pay, columns=["emp_id", "effective_date", "monthly_pay", "grade", "change_type"])
    appr_df = pd.DataFrame(appr, columns=["emp_id", "appraisal_year", "rating"])
    asg_df = pd.DataFrame(assign, columns=["emp_id", "effective_date", "department", "manager_id"])
    return emp, pay_df, appr_df, asg_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--employees", type=int, default=3500)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "raw")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    emp, pay, appr, asg = generate(a.employees, a.seed)
    for name, df in dict(employee_details=emp, pay_history=pay, appraisals=appr, assignments=asg).items():
        df.to_parquet(a.out / f"{name}.parquet", index=False)
    print(f"employee records {len(emp):,} | pay events {len(pay):,} | appraisals {len(appr):,} | assignments {len(asg):,}")
    print(emp.exit_reason.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
