from detecto.core import Model, Dataset
from detecto.utils import read_image
from detecto.visualize import (
    plot_prediction_grid,
    detect_video,
    detect_live,
    show_labeled_image,
)
import time


labels = ["car", "truck", "rock", "tree", "log", "lilypad", "duck"]
model = Model.load("model_weights.pth", labels)

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

image = read_image("screenshot.jpeg")

tic = time.perf_counter()
labels, boxes, scores = model.predict(image)
toc = time.perf_counter()
print(f"Predicted in {toc - tic:0.4f} seconds")

for x in range(10):
    tic = time.perf_counter()
    labels, boxes, scores = model.predict(image)
    toc = time.perf_counter()
    print(f"Predicted in {toc - tic:0.4f} seconds")


show_labeled_image(image, boxes, labels)

# detect_live(model, score_filter=0.7)  # Note: may not work on VMs
