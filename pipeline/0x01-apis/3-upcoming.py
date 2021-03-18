#!/usr/bin/env python3
import requests


def func(e):
    return e["date_unix"]
if __name__ == "__main__":
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = requests.get(url)

    values = r.json()
    values.sort(reverse=True, key=func)
    value = values[0]
    launch_name = value["name"]
    date = value["date_local"]
    rocket_name = value[""]
    print(f"{launch_name} ({date}) <rocket name> - <launchpad name> (<launchpad locality>)")