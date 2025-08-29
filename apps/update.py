import requests
import json


def update(return_request_json):

    with open("data/apps.json", "w", encoding="utf-8") as file:
        apps_json = requests.get(
            "https://raw.githubusercontent.com/JESFTechnology/O.R.I.O.N---Optical-Rendering-for-Innovative-Object-Navigation/refs/heads/main/data/apps.json"
        ).json()
        json.dump(apps_json, file, indent=4, ensure_ascii=False)
    print("data/apps.json atualizado")

    with open("data/apps.json", "r") as file:
        appsUpdate = json.load(file)
        for apps in appsUpdate:
            url_app = (
                "https://raw.githubusercontent.com/JESFTechnology/O.R.I.O.N---Optical-Rendering-for-Innovative-Object-Navigation/refs/heads/main/"
                + apps["path"]
            )
            code_text = requests.get(url_app).text
            with open(apps["path"], "a", encoding="utf-8") as file:
                file.write(code_text)
                print(apps["path"], "atualizado")

    with open("data/config.json", "w", encoding="utf-8") as file:
        json.dump(return_request_json, file, indent=4, ensure_ascii=False)
    print("data/config.json atualizado")

    with open("main.py", "a", encoding="utf-8") as file:
        url_app = "https://raw.githubusercontent.com/JESFTechnology/O.R.I.O.N---Optical-Rendering-for-Innovative-Object-Navigation/refs/heads/main/main.py"
        code_text = requests.get(url_app).text
        file.write(code_text)
    print("main.py atualizado")


def verify():
    return_request_json = requests.get(
        "https://raw.githubusercontent.com/JESFTechnology/O.R.I.O.N---Optical-Rendering-for-Innovative-Object-Navigation/refs/heads/main/data/config.json"
    ).json()
    with open("data/config.json", "r") as file:
        return_file_json = json.load(file)

    if return_request_json["version"] == return_file_json["version"]:
        return False
    print("Atualização necessária")
    update(return_request_json)
    return True
