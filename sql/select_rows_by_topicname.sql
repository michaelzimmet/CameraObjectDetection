SELECT  timestamp,
        data
FROM    messages,
        topics
WHERE   messages.topic_id = topics.id
AND     topics.name = :topic_name
ORDER BY timestamp
;