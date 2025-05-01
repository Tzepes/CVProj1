from enum import Enum
from typing import List, Dict

class Shape(Enum):
    EMPTY = 0
    CIRCLE = 1
    CLOVER = 2
    DIAMOND = 3
    SQUARE = 4
    STAR_4 = 5
    STAR_8 = 6
    
    def __str__(self):
        return str(self.value)

class Color(Enum):
    EMPTY = 'EMP'
    RED = 'R'
    ORANGE = 'O'
    YELLOW = 'Y'
    GREEN = 'G'
    BLUE = 'B'
    WHITE = 'W'
    
    def __str__(self):
        return self.value
    

class Piece:
    def __init__(self, shape: Shape, color: Color, image_patch=None):
        self.shape = shape
        self.color = color
        self.image_patch = image_patch
        
    def img_proprieties(self, image_patch, x_min, y_min, x_max, y_max, line_idx, column_idx):
        self.image_patch = image_patch
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.line_idx = line_idx
        self.column_idx = column_idx
        self.has_x: int = 0 # 0 meaning it does not contain an 'X', 1 meaning it contains an 'X'
        
    def __str__(self):
        return f"{self.shape}{self.color}"
    
class Tile:
    def __init__(self, bonus: int = 0, piece: Piece = None):
        self.piece = piece
        self.image_patch = piece.image_patch if piece else None
        self.bonus = bonus
        self.pieces: List[Piece] = []
    
    def __str__(self):
        return f"{self.piece.shape}{self.piece.color}"
    

class Board:
    def __init__(self):
        # Initialize a 16x16 grid with labeled tiles (A1 to P16)
        self.grid: Dict[str, Tile] = {}
        for i in range(16):  # Rows A to P
            for j in range(16):  # Columns 1 to 16
                label = f"{chr(65 + i)}{j + 1}"  # Generate labels like A1, B1, ..., P16
                self.grid[label] = Tile()
    
    def place_tile(self, position: str, tile: Tile):
        # create self.grid as a matrix of 16x16
        if position in self.grid:
            self.grid[position] = tile
        else:
            raise ValueError(f"Invalid position: {position}")
        
    def display_board(self):
        """
        Display the board in a grid-like format for debugging purposes.
        """
        for i in range(16):  # Rows A to P
            row = []
            for j in range(16):  # Columns 1 to 16
                label = f"{chr(65 + i)}{j + 1}"
                row.append(str(self.grid[label]))
            print(" ".join(row))

    def get_lines(self, new_position: list[str]):
        """
        Get the lines of the board that are affected by the new tile.
        :param new_position: The position of the new tile.
        :return: The lines of the board that are affected by the new tile.
        """
        # Returns the lines (rows/columns) affected by the new tiles
        pass

    # def compute_score(self, positions: List[str]) -> int:


board = Board()
board.display_board()