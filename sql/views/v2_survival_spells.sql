-- =====================================================================
-- V2  Survival spells from a landmark date
-- ---------------------------------------------------------------------
-- Population: employees in post on the landmark date.
-- Follow-up ends at the first of: attrition exit, retirement date, or
-- the end of observation. Only attrition counts as an event; retirement
-- is censored because it is known in advance and handled by rule.
-- =====================================================================
CREATE OR REPLACE VIEW v2_survival_spells AS
WITH pop AS (
    SELECT *
    FROM v1_employee_clean
    WHERE hire_date <= DATE '{landmark}'
      AND (exit_date IS NULL OR exit_date > DATE '{landmark}')
      AND retirement_date > DATE '{landmark}'
),
stop AS (
    SELECT *,
           LEAST(COALESCE(exit_date, DATE '{end_date}'), retirement_date, DATE '{end_date}') AS stop_date
    FROM pop
)
SELECT emp_id, department, birth_date, hire_date,
       DATE_DIFF('day', DATE '{landmark}', stop_date) / 30.44                  AS duration_months,
       CASE WHEN is_attrition = 1 AND exit_date <= DATE '{end_date}'
             AND exit_date <= retirement_date THEN 1 ELSE 0 END               AS event,
       CASE WHEN is_attrition = 1
             AND exit_date <= DATE '{landmark}' + INTERVAL {horizon} MONTH THEN 1 ELSE 0 END AS exit_within_horizon,
       DATE_DIFF('month', hire_date,  DATE '{landmark}')                        AS tenure_months,
       DATE_DIFF('day',   birth_date, DATE '{landmark}') / 365.25               AS age_years
FROM stop;
