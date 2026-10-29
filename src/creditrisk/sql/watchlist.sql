-- Early-warning watchlist: non-standard clients with active triggers, ranked by expected loss.
SELECT ID, grade, pd, action, ews_count, ews_triggers, ead, el,
       limit_action, recommended_limit, reason_1, reason_2, reason_3
FROM (
    SELECT *, ROW_NUMBER() OVER (ORDER BY el DESC) AS el_rank
    FROM scored
    WHERE action <> 'Standard' AND (ews_count >= 1 OR flag = 1)
) ranked
WHERE el_rank <= ?
ORDER BY el DESC;
