#Save the img patches of tiles present inside a folder
import os

# Create a folder to save the tiles if it doesn't exist
output_folder = "shape_dataset"
os.makedirs(output_folder, exist_ok=True)

# Iterate through the tiles_present dictionary
for label, tile_present in tiles_present.items():
    if tile_present:  # Only save images where a tile is detected
        # gray scale patches
        cells_grid[label] = cv2.cvtColor(cells_grid[label], cv2.COLOR_BGR2GRAY)
        output_path = os.path.join(output_folder, f"{label}-5.jpg")
        cv2.imwrite(output_path, cells_grid[label])  # Save the cell image