-- =====================================================================
-- Model base: the five views joined, one row per employee in post on the landmark date
-- =====================================================================
SELECT s.emp_id, s.department, s.duration_months, s.event, s.exit_within_horizon,
       s.tenure_months, s.age_years,
       c.grade, c.pay_vs_grade_median, c.months_since_raise, c.months_since_promotion, c.pay_growth_24m,
       m.mgr_changes_12m, m.transfers_24m, m.months_since_mgr_change,
       p.last_rating, p.rating_change, p.high_performer,
       p.high_performer * CASE WHEN c.pay_vs_grade_median < 1 THEN 1 ELSE 0 END AS underpaid_high_performer
FROM v2_survival_spells s
JOIN v3_compensation c USING (emp_id)
JOIN v4_mobility     m USING (emp_id)
JOIN v5_performance  p USING (emp_id)
ORDER BY s.emp_id   -- deterministic row order (DuckDB is multi-threaded)
