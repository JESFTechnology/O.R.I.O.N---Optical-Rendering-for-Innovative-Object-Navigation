import cv2
import mediapipe as mp
import math
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = tudo, 1 = warnings, 2 = erros, 3 = nada

import modules

WINDOW_NAME = "ORION"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Criar uma janela em tela cheia
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

objeto = False
objects_add = False

projectJson = modules.openProject()
objeto_pos1 = [None]
objeto_pos2 = [None]
objeto_pos3 = [None]
for i in range(len(projectJson["objects"])):
    obj = projectJson["objects"][i]
    x_obj = int(obj["x"])
    y_obj = int(obj["y"])
    if obj["type"] == "Local":
        if obj["name"] == "Cube":
            objeto_pos1.append([(x_obj + obj["size"], y_obj + obj["size"])])
            objeto_pos2.append([(x_obj - obj["size"], y_obj - obj["size"])])
            objeto_pos3.append([(x_obj, y_obj)])

stop = False
index = 0


# print(f"Json aberto: {projectJson}")
def adjustSizeImage(overlay, percent):
    width = int(overlay.shape[1] * percent / 100)
    height = int(overlay.shape[0] * percent / 100)
    overlay = cv2.resize(overlay, (width, height))
    return overlay


def add_overlay(
    frame, path, x_offset=0, y_offset=0, adjustSize=False, adjustPercent=10
):

    overlay = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if adjustSize:
        overlay = adjustSizeImage(overlay, adjustPercent)
    try:
        # Verifica se a sobreposição está no formato correto
        if overlay.shape[2] != 4:
            raise ValueError("A imagem de sobreposição deve estar no formato RGBA.")
        # Assume que a sobreposição está no formato RGBA
        overlay_rgb = overlay[:, :, :3]
        alpha_channel = overlay[:, :, 3] / 255.0  # Normaliza o canal alfa para [0, 1]
        for c in range(3):  # Para cada canal
            frame[
                y_offset : y_offset + overlay.shape[0],
                x_offset : x_offset + overlay.shape[1],
                c,
            ] = (
                frame[
                    y_offset : y_offset + overlay.shape[0],
                    x_offset : x_offset + overlay.shape[1],
                    c,
                ]
                * (1 - alpha_channel)
                + overlay_rgb[:, :, c] * alpha_channel
            )
    except:
        if overlay.ndim != 3 or overlay.shape[2] != 3:
            raise ValueError("A imagem de sobreposição deve estar no formato RGB.")
        # Verifica se a posição de sobreposição está dentro dos limites do quadro
        if (y_offset + overlay.shape[0] > frame.shape[0]) or (
            x_offset + overlay.shape[1] > frame.shape[1]
        ):
            raise ValueError("A sobreposição excede os limites do quadro.")
        # Adiciona a sobreposição diretamente ao quadro
        frame[
            y_offset : y_offset + overlay.shape[0],
            x_offset : x_offset + overlay.shape[1],
        ] = overlay


def distanceFingers(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.hypot(x2 - x1, y2 - y1)


def setup(frame):
    # Carrega imagem com alpha (4 canais)
    add_overlay(
        frame, "images/background.png", 0, 78, adjustSize=True, adjustPercent=66.7
    )
    BLUEBLUEPRINT = (189, 117, 15)
    BLACK = (0, 0, 0)
    cv2.rectangle(frame, (0, 0), (1300, 110), BLACK, -1)
    add_overlay(frame, "images/logo.png", 1090, 120, adjustSize=True, adjustPercent=25)
    add_overlay(frame, "icons/newFile.png", 0, 0, adjustSize=True, adjustPercent=20)
    add_overlay(frame, "icons/openFile.png", 100, 0, adjustSize=True, adjustPercent=20)
    add_overlay(frame, "icons/pen.png", 200, 0, adjustSize=True, adjustPercent=20)
    add_overlay(frame, "icons/erase.png", 300, 0, adjustSize=True, adjustPercent=20)
    add_overlay(frame, "icons/cube.png", 400, 0, adjustSize=True, adjustPercent=20)
    add_overlay(frame, "icons/circle.png", 500, 0, adjustSize=True, adjustPercent=20)
    add_overlay(frame, "icons/power.png", 1170, 5, adjustSize=True, adjustPercent=20)


def loop():
    global stop, index, objeto, objeto_pos1, objeto_pos2, objeto_pos3, projectJson
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,  # 0 mais leve, 1 médio, 2 mais preciso
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9,
    ) as hands:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            res = hands.process(rgb)

            setup(frame)

            if res.multi_hand_landmarks:
                for hand_lms, hand_handedness in zip(
                    res.multi_hand_landmarks, res.multi_handedness
                ):
                    # desenha ossos/pontos (debug visual)
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_style.get_default_hand_landmarks_style(),
                        mp_style.get_default_hand_connections_style(),
                    )

                    thumb = hand_lms.landmark[4]
                    thumb_x = int(thumb.x * w)
                    thumb_y = int(thumb.y * h)

                    index_finger = hand_lms.landmark[8]
                    index_finger_x = int(index_finger.x * w)
                    index_finger_y = int(index_finger.y * h)

                    mid_x = (thumb_x + index_finger_x) // 2
                    mid_y = (thumb_y + index_finger_y) // 2

                    cv2.circle(
                        frame, (index_finger_x, index_finger_y), 8, (0, 255, 0), -1
                    )
                    cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 0), -1)
                    cv2.line(
                        frame,
                        (index_finger_x, index_finger_y),
                        (thumb_x, thumb_y),
                        (0, 255, 0),
                        2,
                    )
                    if (index_finger_x >= 1180 and index_finger_x <= 1250) and (
                        index_finger_y >= 20 and index_finger_y <= 80
                    ):
                        # Stop all code
                        stop = True

                    distance = distanceFingers(thumb, index_finger, w, h)
                    # print("Index: ", index, " Distance: ", distance, " Object: ", objeto)
                    obj = projectJson["objects"][index]
                    if not objeto:
                        if (
                            distance < 40
                            and not objeto
                            and (
                                (mid_x, mid_y) > objeto_pos2[index]
                                and (mid_x, mid_y) < objeto_pos1[index]
                            )
                        ):
                            modules.saveObject(
                                projectJson,
                                index,
                                obj["name"],
                                obj["type"],
                                mid_x,
                                mid_y,
                            )
                            objeto_pos1[index] = (mid_x + 30, mid_y + 30)
                            objeto_pos2[index] = (mid_x - 30, mid_y - 30)
                            objeto = True
                        else:
                            index += 1
                            if index > len(projectJson["objects"]) - 1:
                                index = 0
                    else:
                        distance = distanceFingers(thumb, index_finger, w, h)
                        modules.saveObject(
                            projectJson, index, obj["name"], obj["type"], mid_x, mid_y
                        )
                        objeto_pos1[index] = (mid_x + 30, mid_y + 30)
                        objeto_pos2[index] = (mid_x - 30, mid_y - 30)
                        objeto_pos3[index] = (mid_x, mid_y)
                        if distance > 50 and objeto:
                            objeto = False
                            if obj["type"] == "Local":
                                if obj["name"] == "Cube":
                                    cv2.rectangle(
                                        frame,
                                        objeto_pos1[index],
                                        objeto_pos2[index],
                                        (obj["rgb"][2], obj["rgb"][1], obj["rgb"][0]),
                                        -1,
                                    )
                                elif obj["name"] == "Circle":
                                    cv2.circle(
                                        frame,
                                        objeto_pos3[index],
                                        obj["size"],
                                        (obj["rgb"][2], obj["rgb"][1], obj["rgb"][0]),
                                        -1,
                                    )
                    distance_fingers_x = abs(index_finger_x - thumb_x)
                    distance_fingers_y = abs(index_finger_y - thumb_y)
            for i in range(len(projectJson["objects"])):
                obj = projectJson["objects"][i]
                x_obj = int(obj["x"])
                y_obj = int(obj["y"])
                if obj["type"] == "Local":
                    if obj["name"] == "Cube":
                        # print(f"Desenhando objeto {i}")
                        objeto_pos1[i] = (x_obj + obj["size"], y_obj + obj["size"])
                        objeto_pos2[i] = (x_obj - obj["size"], y_obj - obj["size"])
                        cv2.rectangle(
                            frame,
                            objeto_pos1[i],
                            objeto_pos2[i],
                            (obj["rgb"][2], obj["rgb"][1], obj["rgb"][0]),
                            -1,
                        )
                    elif obj["name"] == "Circle":
                        # print(f"Desenhando objeto {i}")
                        objeto_pos1[i] = (x_obj + obj["size"], y_obj + obj["size"])
                        objeto_pos2[i] = (x_obj - obj["size"], y_obj - obj["size"])
                        cv2.circle(
                            frame,
                            (x_obj, y_obj),
                            obj["size"],
                            (obj["rgb"][2], obj["rgb"][1], obj["rgb"][0]),
                            -1,
                        )
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF == ord("q")) or stop:
                break


def logo_video(VIDEO_PATH):
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Não achei: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Falha ao abrir o vídeo. Verifica codecs/FFmpeg.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 0 else 33  # fallback ~30 FPS

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if False:  # se quiser repetir o vídeo
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord("q")):  # ESC ou q
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        logo_video("videos/inicial.mp4")
        loop()
        #logo_video("videos/final.mp4")
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
