#!/usr/bin/env python3


""" script that displays the upcoming
    launch with these information """

import requests


if __name__ == "__main__":
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = requests.get(url)

    values = r.json()
    values.sort(key=lambda x: x["date_unix"])
    value = values[0]

    # launch and date
    launch_name = value["name"]
    date = value["date_local"]

    # rocket
    rocket_id = value["rocket"]
    r = requests.get(
        "https://api.spacexdata.com/v4/rockets/{}".format(rocket_id))
    rocket_values = r.json()
    rocket_name = rocket_values["name"]

    # launchpad
    launchpad_id = value["launchpad"]
    r = requests.get(
        "https://api.spacexdata.com/v4/launchpads/{}".format(launchpad_id))
    launchpad_values = r.json()
    launchpad_name = launchpad_values["name"]
    launchpad_locality = launchpad_values["locality"]

    print("{} ({}) {} - {} ({})".format(launch_name, date,
                                        rocket_name, launchpad_name,
                                        launchpad_locality))
