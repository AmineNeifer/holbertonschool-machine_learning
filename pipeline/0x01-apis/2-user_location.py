#!/usr/bin/env python3

import sys
import requests


if __name__ == "__main__":
    url = sys.argv[1]
    r = requests.get(url)
    r_json = r.json()
    if r.status_code == 403:
        print("Reset in {} min".format(r.headers["X-Ratelimit-Reset"]))
    elif r.status_code == 200:
        try:
            print(r_json["location"])
        except KeyError:
            print("Not found")
