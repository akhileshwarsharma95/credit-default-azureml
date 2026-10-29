-- Behavioural segments (k-means) profiled on risk and value.
SELECT
    segment_name,
    COUNT(*)                            AS clients,
    COUNT(*) / SUM(COUNT(*)) OVER ()    AS client_share,
    AVG(pd)                             AS mean_pd,
    AVG("default")                      AS observed_dr,
    SUM(ead)                            AS ead,
    SUM(el)                             AS el,
    AVG(util_avg)                       AS mean_utilisation,
    AVG(pay_ratio_avg)                  AS mean_repayment_ratio,
    AVG(n_delays)                       AS mean_late_months,
    AVG(LIMIT_BAL)                      AS mean_limit
FROM scored
GROUP BY segment_name
ORDER BY mean_pd DESC;
