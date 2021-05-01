#!/usr/bin/env python3

"""
Script that prints the location of a specific user, using the GitHub API
"""
import sys
import requests
import time

if __name__ == "__main__":
    url = sys.argv[1]
    r = requests.get(url)
    r_json = r.json()
    if r.status_code == 403:
        limit = r.headers['X-Ratelimit-Reset']
        limit = int((int(limit) - int(time.time())) / 60)
        print('Reset in {} min'.format(limit))
    elif r.status_code == 200:
        try:
            print(r_json["location"])
        except KeyError:
            print("Not found")
    else:
        print("Not found")
