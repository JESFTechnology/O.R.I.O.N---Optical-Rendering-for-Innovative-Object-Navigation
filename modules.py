import json
from time import sleep
import os


def createProject(file_path="projects", name="Blueprint", desc="New Project"):
    arquivos = [
        f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))
    ]
    new_file_path = f"projects/projects_{len(arquivos)}.json"
    if not os.path.exists(new_file_path):
        projectJson = {
            "name": name,
            "desc": desc,
            "objects": [
                {
                    "name": "Cube",
                    "type": "Local",
                    "x": 500,
                    "y": 500,
                    "size": 30,
                    "rgb": [0, 255, 0],
                }
            ],
            "draw": [{"name": "1", "points": [{"x": 0, "y": 0}]}],
        }
        with open(new_file_path, "w", encoding="utf-8") as file:
            json.dump(projectJson, file, indent=4, ensure_ascii=False)
        sleep(1)


def openProject(file_path="projects/projects_0.json"):
    with open(file_path, "r") as file:
        projectJson = json.load(file)
        return projectJson


def saveObject(
    projectJson: dict,
    index: int,
    name="Unknown",
    type="Unknown",
    x: float = 0,
    y: float = 0,
    size: float = 30,
    rgb: list = [0, 255, 0],
    file_path="projects/projects_0.json",
):
    if not projectJson:
        return False
    projectJson["objects"][index]["name"] = name
    projectJson["objects"][index]["type"] = type
    projectJson["objects"][index]["x"] = x
    projectJson["objects"][index]["y"] = y
    projectJson["objects"][index]["size"] = size
    projectJson["objects"][index]["rgb"] = rgb
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(projectJson, file, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erro ao salvar: {e}")
        return False


def coordLocate(projectJson, index):
    data = projectJson["objects"][index]
    x = data["x"]
    y = data["y"]
    return (x, y)


def addCube(projectJson, file_path="projects/projects_0.json"):
    new_object = {
        "name": "Cube",
        "type": "Local",
        "x": 500,
        "y": 500,
        "size": 30,
        "rgb": [0, 255, 0],
    }
    projectJson["objects"].append(new_object)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(projectJson, file, indent=4, ensure_ascii=False)


def addCircle(projectJson, file_path="projects/projects_0.json"):
    new_object = {
        "name": "Circle",
        "type": "Local",
        "x": 500,
        "y": 500,
        "size": 30,
        "rgb": [0, 0, 255],
    }
    projectJson["objects"].append(new_object)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(projectJson, file, indent=4, ensure_ascii=False)


# createProject()
