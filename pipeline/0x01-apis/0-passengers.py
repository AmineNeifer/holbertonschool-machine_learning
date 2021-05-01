#!/usr/bin/env python3


"""contains availableShips function"""
import requests


def availableShips(passengerCount):
    """
    Returns the list of ships that can hold a given
    number of passengers using Swapi API
    """

    lista = []
    url = "https://swapi-api.hbtn.io/api/starships"
    while True:
        r = requests.get(url).json()
        for value in r["results"]:
            try:
                pass_num = int(value["passengers"].replace(',',''))
            except ValueError:
                continue
            if pass_num >= passengerCount:
                lista.append(value["name"])
        url = requests.get(url).json()["next"]
        if (url is None):
            break
    return lista
