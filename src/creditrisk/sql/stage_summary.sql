-- Illustrative IFRS 9 view: exposure, ECL and coverage ratio by stage.
SELECT
    ifrs9_stage                         AS stage,
    COUNT(*)                            AS clients,
    SUM(ead)                            AS ead,
    SUM(ecl)                            AS ecl,
    SUM(ecl) / NULLIF(SUM(ead), 0)      AS coverage_ratio,
    AVG(pd)                             AS mean_pd
FROM scored
GROUP BY ifrs9_stage
ORDER BY stage;
