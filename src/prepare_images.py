import os
import cv2
import numpy as np


def skew(file):
    img = cv2.imread(file)
    scale_percent = 20  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    rows, cols = img.shape[:2]
    # [width (0-1), angle left,  x1], [angle top, heig  ht (0-1), y1]
    M = np.float32([[1, 0.5, 0], [-0.26, 1, 37]])
    img = cv2.warpAffine(img, M, (cols + 147, rows + 37))

    cv2.imwrite(file, img)


path = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "convert")
)

for file in os.listdir(path):
    # skew(file)
    full_path = os.path.join(path, file)
    print(full_path)
    skew(full_path)
