from enum import Enum
from typing import List, Dict

class Shape(Enum):
    CIRCLE = '1'
    CLOVER = '2'
    DIAMOND = '3'
    SQUARE = '4'
    STAR_4 = '5'
    STAR_8 = '6'
    
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
    
    def set_board_config(self, positions):
        bonus_configs = [
            {
                'present_tiles': ['7B', '2G'],
                'bonus_tiles_2': ['2B', '7G'],
                'bonus_tiles_1': ['6B', '7C', '5C', '6D', '4D', '5E', '3E', '4F', '2F', '3G'],
            },
            {
                'present_tiles': ['2B', '7G'],
                'bonus_tiles_2': ['7B', '2G'],
                'bonus_tiles_1': ['3B', '2C', '4C', '3D', '5D', '4E', '6E', '5F', '7F', '6G'],
            },
            {
                'present_tiles': ['10B', '15G'],
                'bonus_tiles_2': ['15B', '10G'],
                'bonus_tiles_1': ['10C', '11B', '11D', '12C', '12E', '13D', '13F', '14E', '14G', '15F'],
            },
            {
                'present_tiles': ['15B', '10G'],
                'bonus_tiles_2': ['10B', '15G'],
                'bonus_tiles_1': ['16B', '15C', '16C', '17D', '18E', '19F', '20G', '15B', '16G', '17F', '18D', '19C', '20B'],
            },
            {
                'present_tiles': ['2J', '7O'],
                'bonus_tiles_2': ['7J', '2O'],
                'bonus_tiles_1': ['3J', '2K', '4K', '3L', '5L', '4M', '6M', '5N', '7N', '60'],
            },
            {
                'present_tiles': ['7J', '2O'],
                'bonus_tiles_2': ['2J', '7O'],
                'bonus_tiles_1': ['6J', '7K', '5K', '6L', '4L', '5M', '3M', '4N', '2N', '3O'],
            },
            {
                'present_tiles': ['15J', '10O'],
                'bonus_tiles_2': ['10J', '15O'],
                'bonus_tiles_1': ['14J', '15K', '13K', '14L', '12L', '13M', '11M', '12N', '10N', '11O'],
            },
            {
                'present_tiles': ['10J', '15O'],
                'bonus_tiles_2': ['15J', '10O'],
                'bonus_tiles_1': ['9J', '10K', '9K', '8L', '7M', '6N', '5O', '10J', '9O', '8N', '7L', '6K', '5J'],
            },
        ]
        
        for config in bonus_configs:
            # Check if there is any overlap between positions and present_tiles
            if set(positions) & set(config['present_tiles']):
                # Update the bonus squares based on the matching configuration
                for tile in config['bonus_tiles_1']:
                    self.bonus_squares[tile] = 1  # Single bonus
                for tile in config['bonus_tiles_2']:
                    self.bonus_squares[tile] = 2  # Double bonus

                print("Bonus squares configured:", self.bonus_squares)

        
    def compute_bonus_1_from_diagonal(self, start: str, end: str) -> list[str]:
        def pos_to_coord(pos: str) -> tuple[int, int]:
            # E.g., '1A' → (0, 0), '3B' → (2, 1)
            row = int(pos[:-1]) - 1  # Extract the number (row) and convert to 0-based index
            col = ord(pos[-1]) - ord('A')  # Extract the character (col) and convert to 0-based index
            return row, col

        def coord_to_pos(row: int, col: int) -> str:
            # E.g., (0, 0) → '1A', (2, 1) → '3B'
            if 0 <= col <= 25 and 0 <= row <= 25:
                return f"{row + 1}{chr(ord('A') + col)}"
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
            row = int(pos[:-1]) - 1  # Extract the number (row) and convert to 0-based index
            col = ord(pos[-1]) - ord('A')  # Extract the character (col) and convert to 0-based index
            return row, col

        def coord_to_pos(row: int, col: int) -> str:
            # E.g., (0, 0) → '1A', (2, 1) → '3B'
            return f"{row + 1}{chr(ord('A') + col)}"

        # Scoring per line formed or extended by the new tiles
        score = 0
        counted_lines = set()
        scored_bonus_tiles = set()
        
        for pos in positions:
            r, c = pos_to_coord(pos)

            for dr, dc in [(0, 1), (1, 0)]:  # horizontal and vertical directions
                line = [(r, c)]

                # Extend backwards
                nr, nc = r - dr, c - dc
                while coord_to_pos(nr, nc) in self.grid:
                    line.insert(0, (nr, nc))
                    nr -= dr
                    nc -= dc

                # Extend forwards
                nr, nc = r + dr, c + dc
                while coord_to_pos(nr, nc) in self.grid:
                    line.append((nr, nc))
                    nr += dr
                    nc += dc

                # Calculate points for the line
                if len(line) > 1:  # Only count lines with more than one tile
                    line_key = tuple(sorted(coord_to_pos(r, c) for r, c in line))
                    if line_key not in counted_lines:
                        counted_lines.add(line_key)
                        score += len(line)  # Add points equal to the number of tiles in the line

            # Add bonus points only if the tile was placed on a bonus square
            if pos in self.bonus_squares and pos not in scored_bonus_tiles:
                bonus_type = self.bonus_squares[pos]
                match bonus_type:
                    case 1:
                        print('scoring 1 ' + str(pos))
                        score += 1  # Single bonus
                    case 2:
                        print('scoring  2 ' + str(pos))  
                        score += 2  # Double bonus
                scored_bonus_tiles.add(pos)  # Double bonus

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
        self.current_player = 1  # Start with Player 1
        self.scores = {1: 0, 2: 0}  # Scores for Player 1 and Player 2

    def switch_player(self):
        """Switch to the other player."""
        self.current_player = 1 if self.current_player == 2 else 2

    def apply_move(self, move: Move):
        for pos, tile in zip(move.positions, move.tiles):
            self.board.place_tile(pos, tile)
        move.score = self.board.compute_score(move.positions)
        self.moves.append(move)

        # Update the current player's score
        self.scores[self.current_player] += move.score

        print('switching player')
        # Switch to the other player
        self.switch_player()
            
