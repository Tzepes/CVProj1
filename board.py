from enum import Enum
from typing import List, Dict

class Shape(Enum):
    CIRCLE = 1
    CLOVER = 2
    DIAMOND = 3
    SQUARE = 4
    STAR_4 = 5
    STAR_8 = 6
    
    def __str__(self):
        return str(self.value)

class Color(Enum):
    RED = 'R'
    ORANGE = 'O'
    YELLOW = 'Y'
    GREEN = 'G'
    BLUE = 'B'
    WHITE = 'W'
    
    def __str__(self):
        return self.value


class Tile:
    def __init__(self, shape: Shape, color: Color):
        self.shape = shape
        self.color = color
        
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

    def __eq__(self, other):
        return self.shape == other.shape and self.color == other.color

    def __hash__(self):
        return hash((self.shape, self.color))
    
    def set_x(self, has_x: int):
        assert has_x == 0 or has_x == 1 # convention 
        self.has_x = has_x

class Board:
    def __init__(self):
        self.grid: Dict[str, Tile] = {} # e.g., {'A1': Tile(...), 'B3': Tile(...)}
        self.bonus_squares: Dict[str, int] = {} #depending on setup (this might be invluenced by what is red from the imabe by CV)
        
    def place_tile(self, position: str, tile: Tile):
        self.grid[position] = tile 
    
    def get_lines(self, new_position: list[str]):
        # Returns the lines (rows/columns) affected by the new tiles
        pass
    
    def compute_score(self, positions: List[str]) -> int:
        def pos_to_coord(pos: str) -> tuple[int, int]:
            # E.g., 'A1' → (0, 0), 'B3' → (1, 2)
            col = ord(pos[0]) - ord('A')
            row = int(pos[1:]) - 1
            return row, col
        
        def coord_to_pos(row: int, col: int) -> str:
            return f"{chr(ord('A') + col)}{row + 1}"
        
        #Scoring per line formed by the new tiles
        score = 0
        counted_lines = set()
        bonus_tiles = set()

        for pos in positions:
            r, c = pos_to_coord(pos)
            
            for dr, dc in [(0, 1), (1, 0)]:  # horizontal and vertical directions
                line = [(r, c)]
                
                #Extend backwards
                nr, nc = r - dr, c - dc
                while coord_to_pos(nr, nc) in self.grid:
                    line.insert(0, (nr, nc))
                    nr -= dr
                    nc -= dc
                
                #Extend forwards
                nr, nc = r + dr, c + dc
                while coord_to_pos(nr, nc) in self.grid:
                    line.append((nr, nc))
                    nr += dr
                    nc += dc
                    
                if len(line) > 2:
                    # Hash to avoid double counting
                    line_key = tuple(sorted(coord_to_pos(r, c) for r, c in line))
                    
                    if line_key not in counted_lines:
                        counted_lines.add(line_key)
                        score += len(line)
                        if len(line) == 6:
                            score += 6  # Qwirkle bonus
                            
            # Bonus square check --- assuming +1 or +2 tiles
            if pos in self.bonus_squares:
                bonus_type = self.bonus_squares[pos]
                match bonus_type:
                    case 1:
                        score += 1  # Double score
                    case 2:
                        score += 2

        #Bonus points are applied once per tile
        for bt in positions:
            score += self.bonus_squares.get(bt, 0)

        # TODO: Add logic for multi-line placement scoring

        return score
    
    
class Move:
    def __init__(self, positions: list[str], tiles: list[Tile]):
        self.positions = positions
        self.tiles = tiles
        self.score = 0
        
    def __str__(self):
        lines = [
            f"{pos}: {tile}" for pos, tile in zip(self.positions, self.tiles)
        ]
        return "\n".join(lines) + f"\nScore: {self.score}"    
        
        
class Game: 
    def __init__(self):
        self.board = Board()
        self.moves: List[Move] = []
        
    def apply_move(self, move:Move):
        for pos, tile in zip(move.positions, move.tiles):
            self.board.place_tile(pos, tile)
        move.score = self.board.compute_score(move.positions)
        self.moves.append(move)
            
            
            
            
game = Game()
move1 = Move(
    positions=["E5", "F5", "G5"],  # Assume horizontal
    tiles=[
        Tile(Shape.CIRCLE, Color.RED),
        Tile(Shape.CLOVER, Color.RED),
        Tile(Shape.DIAMOND, Color.RED),
    ]
)
game.apply_move(move1)
print(move1)