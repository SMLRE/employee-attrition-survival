-- =====================================================================
-- V4  Mobility: manager changes and transfers before the landmark
-- ---------------------------------------------------------------------
-- "People leave managers": a recent manager change is a classic trigger.
-- =====================================================================
CREATE OR REPLACE VIEW v4_mobility AS
WITH a AS (
    SELECT emp_id, effective_date, department, manager_id,
           LAG(department) OVER (PARTITION BY emp_id ORDER BY effective_date) AS prev_dept
    FROM assignments
    WHERE effective_date <= DATE '{landmark}'
)
SELECT s.emp_id,
       COUNT(CASE WHEN a.effective_date > DATE '{landmark}' - INTERVAL 12 MONTH
                   AND a.prev_dept IS NOT NULL THEN 1 END)                          AS mgr_changes_12m,
       COUNT(CASE WHEN a.effective_date > DATE '{landmark}' - INTERVAL 24 MONTH
                   AND a.prev_dept IS NOT NULL AND a.department <> a.prev_dept THEN 1 END) AS transfers_24m,
       DATE_DIFF('month', MAX(a.effective_date), DATE '{landmark}')                  AS months_since_mgr_change
FROM v2_survival_spells s
LEFT JOIN a USING (emp_id)
GROUP BY s.emp_id;
