-- =====================================================================
-- V5  Performance: latest appraisal and its change
-- ---------------------------------------------------------------------
-- Appraisals are annual (January). Only appraisals completed before the
-- landmark are used.
-- =====================================================================
CREATE OR REPLACE VIEW v5_performance AS
WITH r AS (
    SELECT emp_id, appraisal_year, rating,
           ROW_NUMBER() OVER (PARTITION BY emp_id ORDER BY appraisal_year DESC) AS rn
    FROM appraisals
    WHERE appraisal_year <= YEAR(DATE '{landmark}')
)
SELECT s.emp_id,
       MAX(CASE WHEN rn = 1 THEN rating END)                                    AS last_rating,
       MAX(CASE WHEN rn = 1 THEN rating END) - MAX(CASE WHEN rn = 2 THEN rating END) AS rating_change,
       CASE WHEN MAX(CASE WHEN rn = 1 THEN rating END) >= 4 THEN 1 ELSE 0 END   AS high_performer
FROM v2_survival_spells s
LEFT JOIN r USING (emp_id)
GROUP BY s.emp_id;
