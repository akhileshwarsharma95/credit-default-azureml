-- Exposure steering: effect of the recommended limit actions on EAD and expected loss.
SELECT
    limit_action,
    COUNT(*)                AS clients,
    AVG(pd)                 AS mean_pd,
    SUM(LIMIT_BAL)          AS current_limit,
    SUM(recommended_limit)  AS recommended_limit,
    SUM(ead_change)         AS ead_change,
    SUM(el_change)          AS el_change
FROM scored
GROUP BY limit_action
ORDER BY clients DESC;
