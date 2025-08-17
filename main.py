import os
import math
import cv2
import mediapipe as mp
import numpy as np
import modules

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

WINDOW_NAME = "ORION"


# -------------------
# CONFIGURAÇÃO GERAL
# -------------------
class VideoConfig:
    WIDTH = 1920
    HEIGHT = 1080
    FULLSCREEN = True


# -------------------
# HAND DETECTION
# -------------------
class HandDetector:
    def __init__(
        self, max_hands=2, detection_conf=0.9, tracking_conf=0.9, complexity=1
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=complexity,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw(self, frame, hand_lms):
        self.mp_draw.draw_landmarks(
            frame,
            hand_lms,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_style.get_default_hand_landmarks_style(),
            self.mp_style.get_default_hand_connections_style(),
        )


# -------------------
# UTILIDADES
# -------------------
class Utils:
    @staticmethod
    def adjust_size_image(overlay, percent):
        width = int(overlay.shape[1] * percent / 100)
        height = int(overlay.shape[0] * percent / 100)
        return cv2.resize(overlay, (width, height))

    @staticmethod
    def add_overlay(frame, path, x_offset=0, y_offset=0, adjust_size=False, percent=10):
        overlay = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            raise FileNotFoundError(f"Imagem não encontrada: {path}")
        if adjust_size:
            overlay = Utils.adjust_size_image(overlay, percent)

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
            # Overlay sem alpha
            frame[
                y_offset : y_offset + overlay.shape[0],
                x_offset : x_offset + overlay.shape[1],
            ] = overlay

    @staticmethod
    def distance_fingers(p1, p2, w, h):
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        return math.hypot(x2 - x1, y2 - y1)


# -------------------
# OBJETOS DO PROJETO
# -------------------
class ProjectObjects:
    def __init__(self, project_json):
        self.project_json = project_json
        self.objeto = False
        self.index = 0
        # listas de posições
        self.pos1 = [None]
        self.pos2 = [None]
        self.pos3 = [None]
        self.init_positions()

    def init_positions(self):
        for obj in self.project_json["objects"]:
            x_obj, y_obj = int(obj["x"]), int(obj["y"])
            if obj["type"] == "Local" and obj["name"] == "Cube":
                self.pos1.append([(x_obj + obj["size"], y_obj + obj["size"])])
                self.pos2.append([(x_obj - obj["size"], y_obj - obj["size"])])
                self.pos3.append([(x_obj, y_obj)])

    def update_object(self, index, x, y):
        obj = self.project_json["objects"][index]
        modules.saveObject(self.project_json, index, obj["name"], obj["type"], x, y)
        self.pos1[index] = (x + 30, y + 30)
        self.pos2[index] = (x - 30, y - 30)
        self.pos3[index] = (x, y)

    def draw_objects(self, frame):
        for i, obj in enumerate(self.project_json["objects"]):
            x_obj, y_obj = int(obj["x"]), int(obj["y"])
            if obj["type"] == "Local":
                if obj["name"] == "Cube":
                    self.pos1[i] = (x_obj + obj["size"], y_obj + obj["size"])
                    self.pos2[i] = (x_obj - obj["size"], y_obj - obj["size"])
                    cv2.rectangle(
                        frame,
                        self.pos1[i],
                        self.pos2[i],
                        (obj["rgb"][2], obj["rgb"][1], obj["rgb"][0]),
                        -1,
                    )
                elif obj["name"] == "Circle":
                    self.pos1[i] = (x_obj + obj["size"], y_obj + obj["size"])
                    self.pos2[i] = (x_obj - obj["size"], y_obj - obj["size"])
                    cv2.circle(
                        frame,
                        (x_obj, y_obj),
                        obj["size"],
                        (obj["rgb"][2], obj["rgb"][1], obj["rgb"][0]),
                        -1,
                    )


# -------------------
# INTERFACE
# -------------------
class Interface:
    @staticmethod
    def setup(frame):
        Utils.add_overlay(
            frame, "images/background.png", 0, 78, adjust_size=True, percent=66.7
        )
        BLACK = (0, 0, 0)
        cv2.rectangle(frame, (0, 0), (1300, 110), BLACK, -1)
        icons = [
            ("images/logo.png", 1090, 120, 25),
            ("icons/newFile.png", 0, 0, 20),
            ("icons/openFile.png", 100, 0, 20),
            ("icons/pen.png", 200, 0, 20),
            ("icons/erase.png", 300, 0, 20),
            ("icons/cube.png", 400, 0, 20),
            ("icons/circle.png", 500, 0, 20),
            ("icons/power.png", 1170, 5, 20),
        ]
        for path, x, y, percent in icons:
            Utils.add_overlay(frame, path, x, y, adjust_size=True, percent=percent)


# -------------------
# LOOP PRINCIPAL
# -------------------
class MainLoop:
    def __init__(self, video_capture, project_objects, hand_detector):
        self.capture = video_capture
        self.objects = project_objects
        self.hand_detector = hand_detector
        self.stop = False

    def run(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = self.hand_detector.process(frame)
            Interface.setup(frame)

            if res.multi_hand_landmarks:
                for hand_lms in res.multi_hand_landmarks:
                    self.hand_detector.draw(frame, hand_lms)
                    self.handle_fingers(frame, hand_lms, w, h)

            self.objects.draw_objects(frame)
            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF == ord("q")) or self.stop:
                break

    def handle_fingers(self, frame, hand_lms, w, h):
        thumb = hand_lms.landmark[4]
        index_finger = hand_lms.landmark[8]
        thumb_pos = int(thumb.x * w), int(thumb.y * h)
        index_pos = int(index_finger.x * w), int(index_finger.y * h)
        mid_pos = (
            (thumb_pos[0] + index_pos[0]) // 2,
            (thumb_pos[1] + index_pos[1]) // 2,
        )

        cv2.circle(frame, index_pos, 8, (0, 255, 0), -1)
        cv2.circle(frame, thumb_pos, 8, (0, 255, 0), -1)
        cv2.line(frame, index_pos, thumb_pos, (0, 255, 0), 2)

        distance = Utils.distance_fingers(thumb, index_finger, w, h)
        obj = self.objects.project_json["objects"][self.objects.index]

        if not self.objects.objeto:
            if distance < 40 and (
                mid_pos > self.objects.pos2[self.objects.index]
                and mid_pos < self.objects.pos1[self.objects.index]
            ):
                self.objects.update_object(self.objects.index, *mid_pos)
                self.objects.objeto = True
            else:
                self.objects.index += 1
                if self.objects.index >= len(self.objects.project_json["objects"]):
                    self.objects.index = 0
        else:
            self.objects.update_object(self.objects.index, *mid_pos)
            if distance > 50:
                self.objects.objeto = False


# -------------------
# FUNÇÃO DE VÍDEO DE LOGO
# -------------------
def play_video(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Não achei: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Falha ao abrir o vídeo. Verifica codecs/FFmpeg.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps else 33
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(delay) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


# -------------------
# MAIN
# -------------------
if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, VideoConfig.WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VideoConfig.HEIGHT)
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    project_json = modules.openProject()
    project_objects = ProjectObjects(project_json)
    hand_detector = HandDetector()
    main_loop = MainLoop(video_capture, project_objects, hand_detector)

    try:
        play_video("videos/inicial.mp4")
        main_loop.run()
        # play_video("videos/final.mp4")
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
