from queue import PriorityQueue
import cv2
import numpy as np


# 0 is closed
# 2 is open
# 1 is

def main(grid):
    start = (14, 14)
    end = (14, 0)

    newgrid = makeGrid(grid)
    updateGrid(newgrid)

    algorithm(newgrid, newgrid[14][14], newgrid[14][0])

    draw(newgrid)


def findOneItem(grid, item):
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if item == grid[row][col]:
                return (row, col)


class position:  # make a node
    def __init__(self, row, col, state, total_rows):
        self.row = row
        self.col = col
        self.neighbors = []
        self.state = state
        self.total_rows = total_rows

    def get_pos(self):
        return (self.row, self.col)

    def is_closed(self):
        return self.state == 0

    def is_open(self):
        return self.state == 1

    def is_open(self):
        return self.state == 2

    def is_barrier(self):
        return self.state == 3

    def is_duck(self):
        return self.state == 4

    def is_end(self):
        return self.state == 5

    def is_path(self):
        return self.state == 6

    def make_closed(self):
        self.state = 0

    def make_open(self):
        self.state = 2

    def make_path(self):
        self.state = 6

    def update_neghbors(self, grid):
        self.neighbors = []
        # down
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row+1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # up
            self.neighbors.append(grid[self.row-1][self.col])

        # right
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # left
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def makeGrid(textGrid):
    grid = []
    for row in range(len(textGrid)):
        grid.append([])
        for col in range(len(textGrid[row])):
            if textGrid[row][col] == "duck":
                grid[row].append(position(row, col, 4, total_rows=26))
            elif textGrid[row][col] == "":
                grid[row].append(position(row, col, -1, total_rows=26))
            else:
                grid[row].append(position(row, col, 3, total_rows=26))
    return grid


def updateGrid(grid):
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            grid[row][col].update_neghbors(grid)
    return grid


def h(p1, p2):  # hitta hur långt bort den är
    row1, col1 = p1
    row2, col2 = p2
    return abs(row1 - row2) + abs(col1 - col2)


def reconstruct_path(came_from, current):
    while current in came_from:
        current = came_from[current]
        current.make_path()


def algorithm(grid, start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {position: float("inf") for row in grid for position in row}
    g_score[start] = 0
    f_score = {position: float("inf") for row in grid for position in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        current = open_set.get()[2]
        open_set_hash.remove(current)
        if current == end:
            reconstruct_path(came_from, end)
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + \
                    h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        if current != start:
            current.make_closed()
    return False


def draw(grid):
    scale = 10
    frame = np.zeros((29 * scale, 29 * scale, 3), np.uint8)

    for x in range(len(grid)):
        for y in range(len(grid[x])):
            if grid[x][y].is_duck():
                cv2.rectangle(
                    frame,
                    (x * scale + scale, y * scale + scale),
                    (x * scale, y * scale),
                    (255, 255, 255),
                    -1,
                )
            elif grid[x][y].is_barrier():
                cv2.rectangle(
                    frame,
                    (x * scale + scale, y * scale + scale),
                    (x * scale, y * scale),
                    (0, 255, 0),
                    -1,
                )
            elif grid[x][y].is_path():
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
    cv2.namedWindow("grid2", 0)
    cv2.imshow("grid2", frame)
