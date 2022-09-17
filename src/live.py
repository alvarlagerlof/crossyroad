from sys import path
import threading
import time
from importlib.machinery import PathFinder
from math import floor
from typing import final

import cv2
import numpy as np
import scrcpy
from adbutils import adb
from detecto.core import Model

import pathfinder
from prepare_images import skew_frame

final_frame = None
path_grid = None
thinking = False


def detect_live(model):
    client = scrcpy.Client(device="DEVICE SERIAL")
    adb.connect("127.0.0.1:5037")
    # client = scrcpy.Client(device=adb.devices()[0])
    serial = adb.device_list()[0].serial
    if serial is None:
        raise Exception("No adb device found")

    # 216
    client = scrcpy.Client(device=adb.device(serial=serial), max_fps=60, max_width=416)

    client.add_listener(scrcpy.EVENT_FRAME, on_frame)

    client.start()
    while True:
        client.start()


def on_frame(frame):
    if final_frame is not None:
        cv2.namedWindow("final", 0)
        cv2.imshow("final", final_frame)
        cv2.waitKey(1)

    if path_grid is not None:
        cv2.namedWindow("path", 0)
        cv2.imshow("path", path_grid)
        cv2.waitKey(1)

    # If you set non-blocking (default) in constructor, the frame event receiver
    # may receive None to avoid blocking event.
    if frame is not None:
        # frame is an bgr numpy ndarray (cv2' default format)
        # cv2.namedWindow("raw", 0)
        # cv2.imshow("raw", frame)
        # height, width, channels = frame.shape
        # print(height, width, channels)

        if not thinking:
            t1 = threading.Thread(target=heavy, args=[frame])
            t1.start()

        # frame = skew_frame(frame)

        # cv2.namedWindow("skewed", 0)
        # cv2.imshow("skewed", frame)

        # tic = time.perf_counter()
        # predictions = model.predict(frame)

        # render_grid(predictions, score_filter, frame.shape[:2][0], frame.shape[:2][1])

        # # Add the top prediction of each class to the frame
        # for label, box, score in zip(*predictions):
        #     if score < score_filter:
        #         continue

        #     # Since the predictions are for scaled down frames,
        #     # we need to increase the box dimensions
        #     # box *= scale_down_factor  # TODO Issue #16

        #     # Create the box around each object detected
        #     # Parameters: frame, (start_x, start_y), (end_x, end_y), (r, g, b), thickness
        #     cv2.rectangle(
        #         frame,
        #         (int(box[0]), int(box[1])),
        #         (int(box[2]), int(box[3])),
        #         (255, 0, 0),
        #         3,
        #     )

        #     # Write the label and score for the boxes
        #     # Parameters: frame, text, (start_x, start_y), font, font scale, (r, g, b), thickness
        #     cv2.putText(
        #         frame,
        #         "{}: {}".format(label, round(score.item(), 2)),
        #         (int(box[0]), int(box[1] - 10)),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (255, 0, 0),
        #         3,
        #     )

        # cv2.namedWindow("predictions", 0)
        # cv2.imshow("predictions", frame)

        # toc = time.perf_counter()
        # print(f"Predicted and rendered in {toc - tic:0.4f} seconds")

        # cv2.waitKey(1)
        # cv2.destroyAllWindows()


def heavy(frame, score_filter=0.3):
    global final_frame
    global path_grid
    global thinking

    thinking = True

    frame = cv2.blur(frame, (1, 1))

    frame = skew_frame(frame)

    # cv2.namedWindow("skewed", 0)
    # cv2.imshow("skewed", frame)

    tic = time.perf_counter()
    predictions = model.predict(frame)

    render_grid(predictions, score_filter, frame.shape[:2][0], frame.shape[:2][1])

    # Add the top prediction of each class to the frame
    for label, box, score in zip(*predictions):
        if score < score_filter:
            continue

        # Since the predictions are for scaled down frames,
        # we need to increase the box dimensions
        # box *= scale_down_factor  # TODO Issue #16

        # Create the box around each object detected
        # Parameters: frame, (start_x, start_y), (end_x, end_y), (r, g, b), thickness
        cv2.rectangle(
            frame,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 0, 0),
            3,
        )

        # Write the label and score for the boxes
        # Parameters: frame, text, (start_x, start_y), font, font scale, (r, g, b), thickness
        cv2.putText(
            frame,
            "{}: {}".format(label, round(score.item(), 2)),
            (int(box[0]), int(box[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3,
        )

    # cv2.namedWindow("predictions", 0)
    # cv2.imshow("predictions", frame)

    toc = time.perf_counter()
    print(f"Predicted and rendered in {toc - tic:0.4f} seconds")

    final_frame = frame
    # cv2.waitKey(1)
    thinking = False


def render_grid(predictions, score_filter, width, height):
    global path_grid
    tic = time.perf_counter()
    gridsize = 15

    # create basic grid
    grid = np.zeros((29, 29), "<U11")
    grid[14][14] = "duck"

    # scan for duck
    results = [item for item in zip(*predictions) if item[0] == "duck"]
    label, box, score = (None, None, None)

    if len(results) > 0:
        label, box, score = [item for item in zip(*predictions) if item[0] == "duck"][0]
    x_max = None
    y_max = None

    if label != None and score >= score_filter:
        x_max = box[2]  # x max
        y_max = box[3]  # y max

    # create grid frame
    scale = 10
    frame = np.zeros((29 * scale, 29 * scale, 3), np.uint8)

    # scan for items
    if x_max or y_max != None:
        for label, box, score in zip(*predictions):
            if score < score_filter:
                break

            x = floor((box[2].item() - x_max.item()) / width * 14 + 14)
            y = floor((box[3].item() - y_max.item()) / height * 14 + 14)

            # print(label, x, y)
            if grid[x][y] == "":
                if label != "duck":
                    grid[x][y] = label

        # render grid
        for x, col in enumerate(grid):
            for y, row in enumerate(col):
                if row == "duck":
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (255, 255, 255),
                        -1,
                    )
                elif row == "tree":
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (0, 255, 0),
                        -1,
                    )
                elif row == "car":
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (150, 50, 50),
                        -1,
                    )
                elif row == "rock":
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (150, 150, 150),
                        -1,
                    )
                elif row != "":
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (0, 0, 255),
                        -1,
                    )
                else:
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (100, 100, 100),
                        1,
                    )
        # cv2.namedWindow("grid", 0)
        # cv2.imshow("grid", frame)

        path_grid = pathfinder.main(grid)
    # if no duck then render red frame
    else:
        frame[:] = (0, 0, 255)
        # cv2.namedWindow("grid", 0)
        # cv2.imshow("grid", frame)

    toc = time.perf_counter()
    print(f"Renderd grid in in {toc - tic:0.4f} seconds")

    # cv2.waitKey(0)


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
