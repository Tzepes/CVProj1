from enum import Enum
from typing import List, Dict

class Shape(Enum):
    CIRCLE = 'circle'
    CLOVER = 'clover'
    DIAMOND = 'diamond'
    SQUARE = 'square'
    STAR_4 = 'star4'
    STAR_8 = 'star8'
    
    def __str__(self):
        return str(self.value)

class Color(Enum):
    RED = 'red'
    ORANGE = 'orange'
    YELLOW = 'yellow'
    GREEN = 'green'
    BLUE = 'blue'
    WHITE = 'white'
    
    def __str__(self):
        return self.value

#the enums need to be updated to format such as 1, 2, 3, 4, 5, 6 for shape and r, o, y, g, b, w for color
def update_enumTags_to_testFormat(shape: Shape, color: Color) -> tuple[int, int]:
    shape_map = {
        Shape.CIRCLE: 1,
        Shape.CLOVER: 2,
        Shape.DIAMOND: 3,
        Shape.SQUARE: 4,
        Shape.STAR_4: 5,
        Shape.STAR_8: 6
    }
    
    color_map = {
        Color.RED: 'r',
        Color.ORANGE: 'o',
        Color.YELLOW: 'y',
        Color.GREEN: 'g',
        Color.BLUE: 'b',
        Color.WHITE: 'w'
    }
    
    return shape_map[shape], color_map[color]

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
    
    def board_config(self):
        bonus = [
            {
                'present_tiles': ['B7', 'G2'],
                'bonus_tiles_2': ['B2', 'G7'],
                'bonus_tiles_1': ['B6', 'C7', 'C5', 'D6', 'D4', 'E5', 'E3', 'F4', 'F2', 'G3']
            },
            {
                'present_tiles': ['B2', 'G7'],
                'bonus_tiles_2': ['B7', 'G2'],
                'bonus_tiles_1': ['B3', 'C2', 'C4', 'D3', 'D5', 'E4', 'E6', 'F5', 'F7', 'G6']
            },
            {
                'present_tiles': ['B10', 'G15'],
                'bonus_tiles_2': ['B15', 'G10'],
                'bonus_tiles_1': ['B11', 'C10', 'C12', 'D11', 'D13', 'E12', 'E14', 'F13', 'F15', 'G14']
            },
            {
                'present_tiles': ['B15', 'G10'],
                'bonus_tiles_2': ['B10', 'G15'],
                'bonus_tiles_1': ['B14', 'C15', 'C13', 'D14', 'D12', 'E13', 'E11', 'F12', 'F10', 'G11']
            },
            {
                'present_tiles': ['J2', 'O7'],
                'bonus_tiles_2': ['J7', 'O2'],
                'bonus_tiles_1': ['J3', 'K2', 'K4', 'L3', 'L5', 'M4', 'M6', 'N5', 'N7', 'O6']
            },
            {
                'present_tiles': ['J7', 'O2'],
                'bonus_tiles_2': ['J2', 'O7'],
                'bonus_tiles_1': ['J6', 'K7', 'K5', 'L6', 'L4', 'M5', 'M3', 'N4', 'N2', 'O3']
            },
            {
                'present_tiles': ['J15', 'O10'],
                'bonus_tiles_2': ['J10', 'O15'],
                'bonus_tiles_1': ['J14', 'K15', 'K13', 'L14', 'L12', 'M13', 'M11', 'N12', 'N10', 'O11']
            },
            {
                'present_tiles': ['J10', 'O15'],
                'bonus_tiles_2': ['J15', 'O10'],
                'bonus_tiles_1': ['J11', 'K10', 'K12', 'L11', 'L13', 'M12', 'M14', 'N13', 'N15', 'O14']
            }
        ]
        
    def compute_bonus_1_from_diagonal(self, start: str, end: str) -> list[str]:
        def pos_to_coord(pos: str) -> tuple[int, int]:
            col = ord(pos[0]) - ord('A')
            row = int(pos[1:]) - 1
            return row, col

        def coord_to_pos(row: int, col: int) -> str:
            if 0 <= col <= 25 and 0 <= row <= 25:
                return f"{chr(ord('A') + col)}{row + 1}"
            return None  # Out of bounds

        r1, c1 = pos_to_coord(start)
        r2, c2 = pos_to_coord(end)

        dr = 1 if r2 > r1 else -1
        dc = 1 if c2 > c1 else -1

        length = abs(r2 - r1) + 1
        bonus_positions = []

        for i in range(length):
            r = r1 + i * dr
            c = c1 + i * dc

            # Upper diagonal (r - dr, c)
            upper = coord_to_pos(r - dr, c)
            if upper:
                bonus_positions.append(upper)

            # Lower diagonal (r + dr, c)
            lower = coord_to_pos(r + dr, c)
            if lower:
                bonus_positions.append(lower)

        return bonus_positions

    
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
board = Board()
print(board.compute_bonus_1_from_diagonal('B7', 'G2'))
