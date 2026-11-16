-- =====================================================================
-- V1  Clean employee base
-- ---------------------------------------------------------------------
-- * DUPLICATE_RECORD rows are data-quality artefacts -> removed
-- * DEATH is not attrition and not predictable -> removed
-- * RETIREMENT is deterministic (statutory age) -> kept, flagged, and
--   handled as censoring rather than as an event
-- * Attrition event = RESIGNATION + WITHDRAWAL_OF_SERVICE + TERMINATION
-- =====================================================================
CREATE OR REPLACE VIEW v1_employee_clean AS
SELECT emp_id, hire_date, birth_date, department, current_grade, exit_date, exit_reason,
       CASE WHEN exit_reason IN ('RESIGNATION', 'WITHDRAWAL_OF_SERVICE', 'TERMINATION') THEN 1 ELSE 0 END AS is_attrition,
       CASE WHEN exit_reason = 'RETIREMENT' THEN 1 ELSE 0 END                                         AS is_retirement,
       birth_date + INTERVAL 60 YEAR                                                                  AS retirement_date
FROM employee_details
WHERE COALESCE(exit_reason, 'ACTIVE') NOT IN ('DUPLICATE_RECORD', 'DEATH');
