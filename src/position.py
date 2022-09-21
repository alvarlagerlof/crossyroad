class Position:  # make a node
    def __init__(self, row, col, type):
        self.row = row
        self.col = col
        self.neighbors = []
        self.type = type
        self.total_rows = 26
        self.is_path = False

    def get_pos(self):
        return (self.row, self.col)

    def is_blocked(self):
        return self.type in [
            "car",
            "truck",
            "rock",
            "tree",
            "log",
            "rail",
            "train",
            "water",
            "stump",
        ]

    def update_neighbors(self, grid):
        self.neighbors = []
        # down
        if (
            self.row < self.total_rows - 1
            and not grid[self.row + 1][self.col].is_blocked()
        ):
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_blocked():  # up
            self.neighbors.append(grid[self.row - 1][self.col])

        # right
        if (
            self.col < self.total_rows - 1
            and not grid[self.row][self.col + 1].is_blocked()
        ):
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_blocked():  # left
            self.neighbors.append(grid[self.row][self.col - 1])
