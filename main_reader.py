# %pip install opencv-python
# %pip install opencv-python-headless

import cv2
import argparse
from board import Game, Board, Tile, Color, Shape, Move
import numpy as np
import os
import re
from pipelineFunctions import warp_template, align_board, split_board, tile_detection, classify_tiles, classify_tiles_full

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process a folder of images for game moves.")
parser.add_argument(
    "image_folder",
    type=str,
    help="Path to the folder containing images to be processed."
)
args = parser.parse_args()

# Get the folder path from the arguments
image_folder = args.image_folder

# Validate the folder path
if not os.path.isdir(image_folder):
    raise ValueError(f"The provided path '{image_folder}' is not a valid directory.")

# Get the list of images in the folder
image_list = sorted([
    os.path.join(image_folder, f) for f in os.listdir(image_folder)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

if not image_list:
    raise ValueError(f"No valid image files found in the folder '{image_folder}'.")


game = Game()
board = game.board
grid = board.grid
moves = game.moves

already_played_moves = {
    "positions": [],
    "tiles": []
}

output_dir = "moves_output"
os.makedirs(output_dir, exist_ok=True)    

move_count = 0

for img in image_list:
    match = re.search(r'/(?P<number>\d+)_', img) 
    img_num = 0
    if match:
        img_num = match.group('number')
        #set to int
        img_num = int(img_num)
    
    print(f"Processing image: {img}")
    
    warped_img = warp_template(image_list[0])
    
    aligned_img = align_board(warped_img, img, output_size=(800, 800), show_details=False , use_sift=True, homography_threshold=0.85)

    detection_grid, cells_grid = split_board(aligned_img, padding_cell=0, padding_detection=-0.15)
    
    tiles, tiles_present = tile_detection(cells_grid, detection_grid, tile_vs_back_threshol=60, min_foreground_ratio=0.28)

    # Classify tiles and plot classified shapes
    tiles_classified = classify_tiles(tiles_present)
    
    # remove tiles with labels 0, 7, 8, 9 (empty tiles)
    tiles_present = [
        {**tile, 'shape': next((classified_tile['shape'] for classified_tile in tiles_classified if classified_tile['label'] == tile['label']), None)}
        for tile in tiles_present
        if next((classified_tile['shape'] for classified_tile in tiles_classified if classified_tile['label'] == tile['label']), None) not in ["0", "7", "8", "9"]
    ]
    
    tiles_full_classified = classify_tiles_full(tiles_present, tiles_classified)
    # Plot patches with tiles with both shape and color

    new_positions = {
        "positions": [],
        "tiles": []
    }
    
    for tile in tiles_full_classified:
        if tile['label'] not in already_played_moves['positions']:
            # Add the new tile to already_played_moves
            already_played_moves['positions'].append(tile['label'])
            already_played_moves['tiles'].append(Tile(Shape(tile['shape']), Color(tile['color'])))

            # Add the new tile to new_positions
            new_positions['positions'].append(tile['label'])
            new_positions['tiles'].append(Tile(Shape(tile['shape']), Color(tile['color'])))
    
    if move_count == 0:
        board.set_board_config(set(already_played_moves['positions']))
    move = Move(new_positions['positions'], new_positions['tiles'])
    game.apply_move(move)

    print("Move:", move_count)
    
    # print positions and score
    print("Positions:", move.positions)
    print("Score:", move.score)

    if move_count > 0:
        # Save the move to a file
        filename = os.path.join(output_dir, f"1_{move_count:02d}.txt")
        
        # Prepare the content for the file
        move_lines = [f"{pos} {tile}" for pos, tile in zip(move.positions, move.tiles)]
        move_lines.append(str(move.score))  # Add the score as the last line
        
        # Write the content to the file
        with open(filename, "w") as file:
            file.write("\n".join(move_lines))
    
    #destroy current move
    game.moves.pop()
    new_positions['positions'] = []
    new_positions['tiles'] = []
    move_count += 1