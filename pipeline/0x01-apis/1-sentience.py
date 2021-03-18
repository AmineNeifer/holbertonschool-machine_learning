#!/usr/bin/env python3

""" Contains sentientPlanets function"""
import requests


def sentientPlanets():
    """
    Returns the list of names of the home planets of all
    sentient species, using the Swapi API.
    """
    url = "https://swapi-api.hbtn.io/api/species"
    lista = []
    while True:
        r = requests.get(url).json()
        for value in r["results"]:
            if value["designation"] == "sentient":
                try:
                    lista.append(
                        requests.get(
                            value["homeworld"]).json()["name"])
                except BaseException:
                    pass
        url = r["next"]
        if url is None:
            return lista
