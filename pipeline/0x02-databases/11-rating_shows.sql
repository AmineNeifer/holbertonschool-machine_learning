-- lists all shows from hbtn_0d_tvshows_rate by their rating
SELECT tvs.title, SUM(rate) AS rating
FROM tv_shows tvs, tv_show_ratings tvsr
WHERE tvs.id=tvsr.show_id
GROUP BY tvs.id
ORDER BY SUM(rate) DESC;