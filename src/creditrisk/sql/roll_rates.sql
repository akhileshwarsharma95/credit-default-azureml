-- Month-to-month delinquency transitions (roll rates) from the 6-month payment panel.
WITH pairs AS (
    SELECT PAY_6 AS s_from, PAY_5 AS s_to FROM scored UNION ALL
    SELECT PAY_5, PAY_4 FROM scored UNION ALL
    SELECT PAY_4, PAY_3 FROM scored UNION ALL
    SELECT PAY_3, PAY_2 FROM scored UNION ALL
    SELECT PAY_2, PAY_1 FROM scored
),
bucketed AS (
    SELECT
        CASE WHEN s_from <= 0 THEN '0 Current' WHEN s_from = 1 THEN '1 30 dpd'
             WHEN s_from = 2 THEN '2 60 dpd' ELSE '3 90+ dpd' END AS from_bucket,
        CASE WHEN s_to <= 0 THEN '0 Current' WHEN s_to = 1 THEN '1 30 dpd'
             WHEN s_to = 2 THEN '2 60 dpd' ELSE '3 90+ dpd' END AS to_bucket
    FROM pairs
)
SELECT
    from_bucket,
    to_bucket,
    COUNT(*)                                                    AS transitions,
    COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY from_bucket)    AS roll_rate
FROM bucketed
GROUP BY from_bucket, to_bucket
ORDER BY from_bucket, to_bucket;
