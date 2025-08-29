import requests
import json


def verify():
    requests.get(
        "https://github.com/JESFTechnology/O.R.I.O.N---Optical-Rendering-for-Innovative-Object-Navigation/tree/main/data.config"
    ).json()
