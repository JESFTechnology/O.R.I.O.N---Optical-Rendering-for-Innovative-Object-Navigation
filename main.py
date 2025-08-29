import cv2
import subprocess
import numpy as np
import mediapipe as mp
import json
from apps import design, update


class VideoConfig:
    WIDTH = 1920
    HEIGHT = 1080
    FULLSCREEN = True


WINDOW_NAME = "ORION HUB"

codeMenu = 0

update.verify()

with open("data/apps.json", "r") as file:
    appsList = json.load(file)


def run_app(path):
    subprocess.Popen(["python", path])


def adjust_size_image(overlay, percent):
    width = int(overlay.shape[1] * percent / 100)
    height = int(overlay.shape[0] * percent / 100)
    return cv2.resize(overlay, (width, height))


def add_overlay(frame, path, x_offset=0, y_offset=0, adjust_size=False, percent=10):
    overlay = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    if adjust_size:
        overlay = adjust_size_image(overlay, percent)

    # Overlay com alpha
    if overlay.ndim == 3 and overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
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
                * (1 - alpha)
                + overlay[:, :, c] * alpha
            )
    else:
        frame[
            y_offset : y_offset + overlay.shape[0],
            x_offset : x_offset + overlay.shape[1],
        ] = overlay


def menu(frame, finger_x: int = -1, finger_y: int = -1):
    global codeMenu
    match codeMenu:
        case 0:
            # cv2.rectangle(frame, (10, 10), (1270, 710), (170, 178, 32), 3)
            add_overlay(frame, "icons/apps.png", 510, 200, True, 60)
        case 1:
            cv2.rectangle(frame, (10, 10), (1270, 710), (170, 178, 32), 3)
            if len(appsList) < 10:
                space = 20
                y = 0
                i = 0
                for app in appsList:
                    i += 1
                    if i % 3 != 0:
                        add_overlay(
                            frame,
                            app["icon"],
                            space,
                            app["y"] + y,
                            True,
                            app["percent"],
                        )
                        cv2.putText(
                            frame,
                            app["name"],
                            (space, app["y"] + y + 100),
                            cv2.FONT_ITALIC,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                        if finger_x > 0 and finger_y > 0:
                            if (finger_x > space and finger_x < space + 100) and (
                                finger_y > app["y"] and finger_y < app["y"] + y + 100
                            ):
                                codeMenu = i + 2
                        space += 570
                    else:
                        add_overlay(
                            frame,
                            app["icon"],
                            space,
                            app["y"] + y,
                            True,
                            app["percent"],
                        )
                        if not finger_x < 0 and not finger_y < 0:
                            if (finger_x > space and finger_x < space + 30) and (
                                finger_y > app["y"] and finger_y < app["y"] + 100
                            ):
                                cv2.putText(
                                    frame,
                                    f"{app["name"]}",
                                    (20, 200),
                                    cv2.FONT_ITALIC,
                                    1,
                                    (255, 255, 255),
                                    1,
                                )
                        else:
                            cv2.putText(
                                frame,
                                "Dedo não encontrado",
                                (20, 200),
                                cv2.FONT_ITALIC,
                                1,
                                (255, 255, 255),
                                1,
                            )
                        space = 20
                        y += 290
                i = 0
            # elif len(appsList) < 25:
            #    space = 20
            #    y = 0
            #    i = 0
            #    for app in appsList:
            #        i += 1
            #        if i % 3 != 0:
            #            add_overlay(
            #                frame,
            #                app["icon"],
            #                space,
            #                app["y"] + y,
            #                True,
            #                app["percent"],
            #            )
            #            space += 285
            #        else:
            #            add_overlay(
            #                frame,
            #                app["icon"],
            #                space,
            #                app["y"] + y,
            #                True,
            #                app["percent"],
            #            )
            #            space = 20
            #            y += 100
            #    i = 0
        case 2:
            # design.start()
            codeMenu = 0
        case 3:
            design.start()
            codeMenu = 0
        case _:
            codeMenu = 0


def main():
    global codeMenu

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VideoConfig.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VideoConfig.HEIGHT)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # espelhado
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Fundo preto
        # frame = cv2.imread("images/fundoPreto.png")
        if frame is None:
            frame = 255 * np.ones((1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(
            frame, (0, 0), (VideoConfig.WIDTH, VideoConfig.HEIGHT), (0, 0, 0), -1
        )
        if results.multi_hand_landmarks:
            for handLms, handType in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handType.classification[0].label  # "Left" ou "Right"
                if label == "Right":  # só mão direita
                    x = int(handLms.landmark[8].x * w)  # dedo indicador
                    y = int(handLms.landmark[8].y * h)

                    # desenhar ponto
                    cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                    # checa se tá no centro da tela
                    cv2.putText(
                        frame,
                        f"{x} , {y}",
                        (20, 700),
                        cv2.FONT_ITALIC,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    if (
                        w // 3 < x < 2 * w // 3
                        and h // 3 < y < 2 * h // 3
                        and codeMenu != 1
                    ):
                        codeMenu = 1
                        menu(frame)
                    else:
                        menu(frame, finger_x=x, finger_y=y)

                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        menu(frame)
        # Mostra as janelas
        cv2.imshow(WINDOW_NAME, frame)
        # cv2.imshow("Camera", cam_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
