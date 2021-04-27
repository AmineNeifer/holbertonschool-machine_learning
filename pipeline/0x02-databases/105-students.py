#!/usr/bin/env python3
""" Contains top_students function"""


def top_students(mongo_collection):
    """returns all students sorted by average score"""
    docs = []
    for doc in mongo_collection.find():
        score_sum = 0
        for d in doc["topics"]:
            score_sum += d['score']
        doc['averageScore'] = score_sum / len(doc["topics"])
        docs.append(doc)
    return (sorted(docs, key=lambda i: i['averageScore'], reverse=True))
