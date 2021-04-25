-- lists all genres in the database hbtn_0d_tvshows_rate by their rating
SELECT tvg.name, SUM(rate) AS rating
FROM tv_genres tvg, tv_show_genres tvsg, tv_show_ratings tvsr
WHERE tvg.id=tvsg.genre_id AND tvsg.show_id=tvsr.show_id
GROUP BY tvg.name
ORDER BY SUM(rate) DESC;