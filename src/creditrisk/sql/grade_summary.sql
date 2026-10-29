-- Rating-grade view: where the volume, the exposure and the expected loss sit.
WITH g AS (
    SELECT
        grade,
        COUNT(*)         AS clients,
        AVG(pd)          AS mean_pd,
        AVG("default")   AS observed_dr,
        SUM(ead)         AS ead,
        SUM(el)          AS el,
        AVG(util_1)      AS mean_utilisation
    FROM scored
    GROUP BY grade
)
SELECT
    grade,
    clients,
    clients / SUM(clients) OVER ()  AS client_share,
    mean_pd,
    observed_dr,
    ead,
    ead / SUM(ead) OVER ()          AS ead_share,
    el,
    el / SUM(el) OVER ()            AS el_share,
    el / NULLIF(ead, 0)             AS el_rate,
    mean_utilisation
FROM g
ORDER BY grade;
