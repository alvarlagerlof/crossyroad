from heapq import heapify
from turtle import position, st
import numpy as np
import cv2
from math import floor
from queue import PriorityQueue
import operator
from position import Position
from scrcpy.const import (
    KEYCODE_DPAD_UP,
    KEYCODE_DPAD_DOWN,
    KEYCODE_DPAD_LEFT,
    KEYCODE_DPAD_RIGHT,
    KEYCODE_DPAD_CENTER,
)


class Grid:
    def __init__(self, predictions, score_filter, width, height):
        self.width = width
        self.height = height
        self.predictions = predictions
        self.score_filter = score_filter

        # # scan for duck
        # results = [item for item in zip(*self.predictions) if item[0] == "duck"]
        # label, box, score = (None, None, None)

        # if len(results) > 0:
        #     label, box, score = [
        #         item for item in zip(*self.predictions) if item[0] == "duck"
        #     ][0]
        # x_max = None
        # y_max = None

        # if label != None and score >= score_filter:
        #     x_max = box[2]  # x max
        #     y_max = box[3]  # y max
        # print(x_max, y_max)

        x, y = self.find_duck()
        self.make_grid(x, y)
        print(x, y)

    def find_duck(self):
        results = [item for item in zip(*self.predictions) if item[0] == "duck"]
        label, box, score = (None, None, None)

        if len(results) > 0:
            label, box, score = [
                item for item in zip(*self.predictions) if item[0] == "duck"
            ][0]
        x = None
        y = None

        if label != None and score >= self.score_filter:
            x = box[2]  # x max
            y = box[3]  # y max

        return (x, y)

    def make_grid(self, duck_x, duck_y):
        self.grid = []
        for row in range(29):
            self.grid.append([])
            for col in range(29):
                self.grid[row].append(None)

        # self.grid = np.zeros((29, 29), "<U11")

        if duck_x or duck_y != None:
            for label, box, score in zip(*self.predictions):
                if score < self.score_filter:
                    break

                x = floor((box[2].item() - duck_x.item()) / self.width * 14 + 14)
                y = floor((box[3].item() - duck_y.item()) / self.height * 14 + 14)

                if self.grid[x][y] == None:
                    self.grid[x][y] = Position(x, y, label)

            for row in range(len(self.grid)):
                for col in range(len(self.grid[row])):
                    if self.grid[row][col] == None:
                        self.grid[row][col] = Position(row, col, "empty")

            for row in range(len(self.grid)):
                for col in range(len(self.grid[row])):
                    self.grid[row][col].update_neighbors(self.grid)

    def a_star(self):
        def distance(p1, p2):  # hitta hur långt bort den är
            row1, col1 = p1
            row2, col2 = p2
            return abs(row1 - row2) + abs(col1 - col2)

        if self.grid[14][14] != None and self.grid[14][14].type == "duck":
            count = 0
            start = self.grid[14][14]
            end = self.grid[14][0]

            open_set = PriorityQueue()
            open_set.put((0, count, start))
            came_from = {}
            g_score = {position: float("inf") for row in self.grid for position in row}
            g_score[start] = 0
            f_score = {position: float("inf") for row in self.grid for position in row}
            f_score[start] = distance(start.get_pos(), end.get_pos())

            open_set_hash = {start}

            while not open_set.empty():
                current = open_set.get()[2]
                open_set_hash.remove(current)
                if current == end:
                    while end in came_from:
                        x = end
                        end = came_from[end]
                        end.is_path = True
                    print(x.get_pos(), start.get_pos())
                    res = tuple(map(operator.sub, x.get_pos(), start.get_pos()))
                    print(res)
                    if res == (0, 0):
                        return None
                    if res == (0, 1):
                        return KEYCODE_DPAD_DOWN
                    if res == (1, 0):
                        return KEYCODE_DPAD_RIGHT
                    if res == (-1, 0):
                        return KEYCODE_DPAD_LEFT
                    if res == (0, -1):
                        return KEYCODE_DPAD_UP

                for neighbor in current.neighbors:
                    temp_g_score = g_score[current] + 1
                    if temp_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = temp_g_score
                        f_score[neighbor] = temp_g_score + distance(
                            neighbor.get_pos(), end.get_pos()
                        )
                        if neighbor not in open_set_hash:
                            count += 1
                            open_set.put((f_score[neighbor], count, neighbor))
                            open_set_hash.add(neighbor)
                            # neighbor.make_open()

                # if current != start:
                # current.make_closed()
            return False
        return False

    def render(self):
        scale = 10
        frame = np.zeros((29 * scale, 29 * scale, 3), np.uint8)

        for x in range(len(self.grid)):
            for y in range(len(self.grid[x])):
                if isinstance(self.grid[x][y], str) or self.grid[x][y] == None:
                    break

                if self.grid[x][y].type == "duck":
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (255, 255, 255),
                        -1,
                    )
                elif self.grid[x][y].is_blocked():
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (0, 255, 0),
                        -1,
                    )
                elif self.grid[x][y].is_path:
                    cv2.rectangle(
                        frame,
                        (x * scale + scale, y * scale + scale),
                        (x * scale, y * scale),
                        (150, 50, 50),
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

        return frame
