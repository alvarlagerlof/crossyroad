import threading
import time
from math import floor
import cv2
import numpy as np
import scrcpy
from adbutils import adb
from detecto.core import Model
from grid import Grid
import pathfinder
from prepare_images import skew_frame

skewed_frame = None
grid_frame = None
thinking = False
direction = None
device = None


def detect_live(model):
    global device
    adb.connect("127.0.0.1:5037")
    serial = adb.device_list()[0].serial
    if serial is None:
        raise Exception("No adb device found")

    # Setup
    device = adb.device(serial=serial)
    client = scrcpy.Client(device=device, max_fps=60, max_width=416)

    client.add_listener(scrcpy.EVENT_FRAME, on_frame)

    client.start()
    while True:
        client.start()


def on_frame(frame):
    if skewed_frame is not None:
        cv2.namedWindow("skewed", 0)
        cv2.imshow("skewed", skewed_frame)
        cv2.waitKey(1)

    if grid_frame is not None:
        cv2.namedWindow("grid", 0)
        cv2.imshow("grid", grid_frame)
        cv2.waitKey(1)

    # If you set non-blocking (default) in constructor, the frame event receiver
    # may receive None to avoid blocking event.
    if frame is not None:
        if not thinking:
            print("pressed ", direction)

            device.shell("input keyevent " + str(direction))

            t1 = threading.Thread(target=heavy, args=[frame])
            t1.start()


def heavy(frame, score_filter=0.3):
    global thinking
    global grid_frame
    global skewed_frame
    global direction
    thinking = True
    frame = cv2.blur(frame, (1, 1))
    skewed_frame = skew_frame(frame)
    predictions = model.predict(skewed_frame)
    grid = Grid(
        predictions, score_filter, skewed_frame.shape[:2][0], skewed_frame.shape[:2][1]
    )
    direction = grid.a_star()
    grid_frame = grid.render()

    thinking = False


labels = [
    "car",
    "truck",
    "rock",
    "tree",
    "log",
    "lilypad",
    "duck",
    "rail",
    "train",
    "water",
    "light off",
    "light on",
    "coin",
    "stump",
]


model = Model.load("../tmp/model_weights.pth", labels)
detect_live(model)
