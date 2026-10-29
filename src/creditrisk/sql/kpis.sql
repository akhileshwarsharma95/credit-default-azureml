-- Portfolio headline KPIs. "default" is NULL when labels are not (yet) known.
SELECT
    COUNT(*)                                                     AS clients,
    SUM(ead)                                                     AS total_ead,
    SUM(el)                                                      AS total_el,
    SUM(el) / NULLIF(SUM(ead), 0)                                AS el_rate,
    SUM(ecl)                                                     AS total_ecl,
    AVG(pd)                                                      AS mean_pd,
    AVG(flag)                                                    AS intervene_rate,
    AVG(CASE WHEN action = 'Watch' THEN 1 ELSE 0 END)            AS watch_rate,
    SUM(CASE WHEN ifrs9_stage = 2 THEN ead ELSE 0 END) / NULLIF(SUM(ead), 0) AS stage2_ead_share,
    SUM(CASE WHEN ifrs9_stage = 3 THEN ead ELSE 0 END) / NULLIF(SUM(ead), 0) AS stage3_ead_share,
    AVG(CAST(ews_count >= 2 AS INTEGER))                         AS ews_multi_trigger_rate,
    AVG("default")                                               AS observed_dr
FROM scored;
