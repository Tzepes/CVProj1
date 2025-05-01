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
        if self.bonus > 0:
            return f"+{self.bonus}"
        elif self.piece is None:
            return "EMPTY"  # Default representation for an empty tile
        return f"{self.piece.shape}{self.piece.color}"
    

class Board:
    def __init__(self):
        # Initialize a 16x16 grid with labeled tiles (A1 to P16)
        # and populate with tiles with default values
        self.tile_list: Dict[str, Tile] = {}
        self.grid_matrix: List[List[Tile]] = [[None for _ in range(16)] for _ in range(16)]
        for i in range(16):  # Rows A to P
            for j in range(16):
                label = f"{chr(65 + i)}{j + 1}"
                self.tile_list[label] = Tile()
                self.grid_matrix[i][j] = self.tile_list[label]                        
    
    def place_piece(self, position: str, piece: Piece):
        """
        Place a piece on the board at the specified position.
        :param position: The position on the board (e.g., "A1").
        :param piece: The piece to place on the board.
        """
        if position in self.tile_list:
            self.tile_list[position].piece = piece
            self.tile_list[position].image_patch = piece.image_patch
            self.tile_list[position].pieces.append(piece)
        else:
            raise ValueError(f"Invalid position: {position}")
        
    def display_board(self, show_position_tags: bool = True):
        """
        Display the board using self.tile_list in a grid-like format for debugging purposes.
        """
        for i in range(16):
            row = []
            for j in range(16):
                if show_position_tags:
                    label = f"{chr(65 + i)}{j + 1}"
                    row.append(label + ' ' + str(self.tile_list[label]))
                else:
                    label = f"{chr(65 + i)}{j + 1}"  # Construct the key (e.g., "A1", "B2")
                    row.append(str(self.tile_list[label]))  # Get the tile representation
            print(" | ".join(row))

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
# board.display_board(show_position_tags=False)
print(board.grid_matrix[0][0])