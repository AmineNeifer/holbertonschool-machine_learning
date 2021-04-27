#!/usr/bin/env python3
""" Improved 34-log_stats.py by adding the top 10 of the most
present IPs in the collection nginx of the database logs"""
from pymongo import MongoClient
from bson.son import SON

if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.logs.nginx
    method = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    print('{} logs'.format(school_collection.count_documents({})))
    print('Methods:')
    for meth in method:
        print('\tmethod {}: {}'.format(
            meth, school_collection.count_documents({'method': meth})))
    print('{} status check'.format(school_collection.count_documents(
        {'method': 'GET', 'path': '/status'})))

    pipeline = [
        {"$group": {"_id": "$ip", "count": {"$sum": 1}}},
        {"$sort": SON([("count", -1)])}
    ]

    res = school_collection.aggregate(pipeline=pipeline)
    print("IPs")
    for item in list(res)[:10]:
        print('\t{}: {}'.format(item['_id'], item['count']))
