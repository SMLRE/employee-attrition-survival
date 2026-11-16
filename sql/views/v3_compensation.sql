-- =====================================================================
-- V3  Compensation as of the landmark date
-- ---------------------------------------------------------------------
-- Pay relative to the grade median is a stronger signal than absolute pay:
-- people compare themselves with peers in the same grade.
-- =====================================================================
CREATE OR REPLACE VIEW v3_compensation AS
WITH hist AS (
    SELECT * FROM pay_history WHERE effective_date <= DATE '{landmark}'
),
latest AS (
    SELECT emp_id, monthly_pay, grade,
           ROW_NUMBER() OVER (PARTITION BY emp_id ORDER BY effective_date DESC, monthly_pay DESC) AS rn  -- raise + promotion can share a date
    FROM hist
),
cur AS (SELECT emp_id, monthly_pay, grade FROM latest WHERE rn = 1),
grade_med AS (
    SELECT grade, MEDIAN(monthly_pay) AS grade_median_pay
    FROM cur JOIN v2_survival_spells USING (emp_id) GROUP BY grade
),
events AS (
    SELECT emp_id,
           MAX(CASE WHEN change_type IN ('RAISE', 'PROMOTION') THEN effective_date END) AS last_raise_date,
           MAX(CASE WHEN change_type = 'PROMOTION' THEN effective_date END)              AS last_promo_date,
           MAX(CASE WHEN effective_date <= DATE '{landmark}' - INTERVAL 24 MONTH THEN monthly_pay END) AS pay_24m_ago
    FROM hist GROUP BY emp_id
)
SELECT c.emp_id, c.grade, c.monthly_pay,
       c.monthly_pay / g.grade_median_pay                                                    AS pay_vs_grade_median,
       DATE_DIFF('month', COALESCE(e.last_raise_date, s.hire_date), DATE '{landmark}')       AS months_since_raise,
       DATE_DIFF('month', COALESCE(e.last_promo_date, s.hire_date), DATE '{landmark}')       AS months_since_promotion,
       LN(c.monthly_pay / COALESCE(e.pay_24m_ago, c.monthly_pay))                            AS pay_growth_24m
FROM cur c
JOIN grade_med g USING (grade)
JOIN events e USING (emp_id)
JOIN v2_survival_spells s USING (emp_id);
