#!/usr/bin/env python3

import requests

def sentientPlanets():
    url = "https://swapi-api.hbtn.io/api/species"
    lista = []
    while True:        
        r = requests.get(url).json()
        for value in r["results"]:
            if value["designation"] == "sentient":
                try: 
                    lista.append(requests.get(value["homeworld"]).json()["name"])
                except:
                    pass
        url = r["next"]
        if url is None:
            return lista