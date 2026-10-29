-- Monthly portfolio trend (Apr -> Sep): delinquency rate and utilisation.
SELECT * FROM (
    SELECT 1 AS month_idx, 'Apr' AS month, AVG(CAST(PAY_6 >= 1 AS INT)) AS delinquency_rate, SUM(BILL_AMT6) / SUM(LIMIT_BAL) AS utilisation FROM scored UNION ALL
    SELECT 2, 'May', AVG(CAST(PAY_5 >= 1 AS INT)), SUM(BILL_AMT5) / SUM(LIMIT_BAL) FROM scored UNION ALL
    SELECT 3, 'Jun', AVG(CAST(PAY_4 >= 1 AS INT)), SUM(BILL_AMT4) / SUM(LIMIT_BAL) FROM scored UNION ALL
    SELECT 4, 'Jul', AVG(CAST(PAY_3 >= 1 AS INT)), SUM(BILL_AMT3) / SUM(LIMIT_BAL) FROM scored UNION ALL
    SELECT 5, 'Aug', AVG(CAST(PAY_2 >= 1 AS INT)), SUM(BILL_AMT2) / SUM(LIMIT_BAL) FROM scored UNION ALL
    SELECT 6, 'Sep', AVG(CAST(PAY_1 >= 1 AS INT)), SUM(BILL_AMT1) / SUM(LIMIT_BAL) FROM scored
) t
ORDER BY month_idx;
