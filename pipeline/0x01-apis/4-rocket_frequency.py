#!/usr/bin/env python3

"""
displays the number of launches per rocket.
"""
import requests

if __name__ == "__main__":
    launches = {}
    url = "https://api.spacexdata.com/v4/launches/"
    r = requests.get(url)

    values = r.json()
    values.sort(key=lambda e: e["date_unix"])

    for value in values:
        rocket_id = value["rocket"]
        if rocket_id in launches:
            launches[rocket_id] += 1
        else:
            launches[rocket_id] = 1
    for rocket_id in launches:
        r = requests.get(
            "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id))
        rocket_values = r.json()
        rocket_name = rocket_values["name"]
        launches[rocket_name] = launches.pop(rocket_id)

    for k, v in sorted(sorted(launches.items()),
                       key=lambda x: x[1], reverse=True):
        print(f"{k}: {v}")
