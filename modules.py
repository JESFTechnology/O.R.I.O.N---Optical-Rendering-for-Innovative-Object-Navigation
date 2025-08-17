import json
from time import sleep


def openProject():
    with open("projects/projects.json", "r") as file:
        projectJson = json.load(file)
        return projectJson


def saveObject(
    projectJson: dict,
    index: int,
    name="Unknown",
    type="Unknown",
    x: float = 0,
    y: float = 0,
):
    if not projectJson:
        return False
    projectJson["objects"][index]["name"] = name
    projectJson["objects"][index]["type"] = type
    projectJson["objects"][index]["x"] = x
    projectJson["objects"][index]["y"] = y
    try:
        with open("projects/projects.json", "w", encoding="utf-8") as file:
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


#projectJson = openProject()
#sleep(1)
#print(saveObject(projectJson, 0, "Cube", "Local", 0, 0))
# projectJsonObjects = projectJson["objects"]
# for i in range(len(projectJsonObjects), 0, -1):
#    listObjects = projectJsonObjects[i-1]
#    print(listObjects["name"], end=" ")
#    print(listObjects["type"], end=" ")
#    print(float(listObjects["x"]), end=" ")
#    print(float(listObjects["y"]))
