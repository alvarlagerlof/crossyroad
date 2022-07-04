from detecto.core import Model, Dataset
from detecto.utils import read_image
from detecto.visualize import (
    plot_prediction_grid,
    detect_video,
    detect_live,
    show_labeled_image,
)
import time


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
#
# image = read_image("screenshot.jpeg")
# predictions = model.predict(image)

# val_dataset = Dataset("val_labels.csv", "data/validation/")

# images = []
# for i in range(3):
#     image, _ = val_dataset[i]
#     images.append(image)

# top_predictions = model.predict_top(images)

# print(predictions)
# print(top_predictions)

detect_video(model, "../tmp/skewed3.avi", "../tmp/detected2.avi", score_filter=0.5)


# image = read_image("screenshot.jpeg")
#
# tic = time.perf_counter()
# labels, boxes, scores = model.predict(image)
# toc = time.perf_counter()
# print(f"Predicted in {toc - tic:0.4f} seconds")
#
# plot_prediction_grid(model, [image], dim=(1,1), figsize=(1,1), score_filter=0.6 )
# show_labeled_image(image, boxes, labels)


# detect_live(model, score_filter=0.7)  # Note: may not work on VMs
