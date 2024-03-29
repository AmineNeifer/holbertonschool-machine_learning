#!/usr/bin/env python3
""" Contains insert_school function"""


def insert_school(mongo_collection, **kwargs):
    """ inserts a new document in a collection based on kwargs"""
    returned_value = mongo_collection.insert_one(kwargs).inserted_id
    return returned_value
