import cv2
import numpy as np
from detecto.core import Model
import time


def detect_video(model, input_file, output_file, fps=30, score_filter=0.6):
    """Takes in a video and produces an output video with object detection
    run on it (i.e. displays boxes around detected objects in real-time).
    Output videos should have the .avi file extension. Note: some apps,
    such as macOS's QuickTime Player, have difficulty viewing these
    output videos. It's recommended that you download and use
    `VLC <https://www.videolan.org/vlc/index.html>`_ if this occurs.


    :param model: The trained model with which to run object detection.
    :type model: detecto.core.Model
    :param input_file: The path to the input video.
    :type input_file: str
    :param output_file: The name of the output file. Should have a .avi
        file extension.
    :type output_file: str
    :param fps: (Optional) Frames per second of the output video.
        Defaults to 30.
    :type fps: int
    :param score_filter: (Optional) Minimum score required to show a
        prediction. Defaults to 0.6.
    :type score_filter: float

    **Example**::

        >>> from detecto.core import Model
        >>> from detecto.visualize import detect_video

        >>> model = Model.load('model_weights.pth', ['tick', 'gate'])
        >>> detect_video(model, 'input_vid.mp4', 'output_vid.avi', score_filter=0.7)
    """

    # Read in the video
    video = cv2.VideoCapture(input_file)

    # Video frame dimensions
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Scale down frames when passing into model for faster speeds
    # scaled_size = 800
    # scale_down_factor = min(frame_height, frame_width) / scaled_size

    # The VideoWriter with which we'll write our video with the boxes and labels
    # Parameters: filename, fourcc, fps, frame_size
    out = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"DIVX"), fps, (frame_width, frame_height)
    )

    # Transform to apply on individual frames of the video
    # transform_frame = transforms.Compose(
    #     [  # TODO Issue #16
    #         transforms.ToPILImage(),
    #         transforms.Resize(scaled_size),
    #         transforms.ToTensor(),
    #         normalize_transform(),
    #     ]
    # )

    # Loop through every frame of the video
    count = 0

    while True:
        count += 1
        print("frame", count)

        ret, frame = video.read()
        # Stop the loop when we're done with the video
        if not ret:
            break

        # tic = time.perf_counter()

        scale_percent = 20  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        rows, cols = frame.shape[:2]
        # [width (0-1), angle left,  x1], [angle top, heig  ht (0-1), y1]
        M = np.float32([[1, 0.5, 0], [-0.26, 1, 58]])
        frame = cv2.warpAffine(frame, M, (cols + 222, rows + 58))

        # toc = time.perf_counter()
        # print(f"Resized and warped in {toc - tic:0.4f} seconds")

        # cv2.namedWindow("warp", 0)
        # cv2.imshow("warp", frame)

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

        cv2.namedWindow("predictions", 0)
        cv2.imshow("predictions", frame)

        # Write this frame to our video file
        out.write(frame)
        toc = time.perf_counter()
        print(f"Predicted and wrote in {toc - tic:0.4f} seconds")

        # If the 'q' key is pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # cv2.waitKey(0)

    # When finished, release the video capture and writer objects
    video.release()
    out.release()

    # Close all the frames
    cv2.destroyAllWindows()


def render_grid(predictions, score_filter, width, height):
    tic = time.perf_counter()

    # create basic grid
    grid = np.zeros((21, 21), "<U11")
    grid[10][10] = "duck"

    # scan for duck
    label, box, score = [item for item in zip(*predictions) if item[0] == "duck"][0]
    x_max = None
    y_max = None

    if label != None and score >= score_filter:
        x_max = box[2]  # x max
        y_max = box[3]  # y max

    # create grid frame
    scale = 10
    frame = np.zeros((21 * scale, 21 * scale, 3), np.uint8)

    # scan for items
    if x_max or y_max != None:
        for label, box, score in zip(*predictions):
            if score < score_filter:
                break

            x = round((box[2].item() - x_max.item()) / width * 10 + 10)
            y = round((box[3].item() - y_max.item()) / height * 10 + 10)

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
                elif row != "":
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (150, 150, 150),
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

        cv2.namedWindow("grid", 0)
        cv2.imshow("grid", frame)

    else:
        frame[:] = (0, 0, 255)
        cv2.namedWindow("grid", 0)
        cv2.imshow("grid", frame)

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


model = Model.load("model_weights.pth", labels)

# detect_video(model, "../tmp/input.mp4", "../tmp/detected.avi", score_filter=0.3)
detect_video(model, "../tmp/input.mp4", "../tmp/detected_short.mp4", score_filter=0.2)
