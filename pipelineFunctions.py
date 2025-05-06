import numpy as np
from imageProcessing import detect_board_and_warp, split_board_into_cells
from imgAlignment import align_board_sift_orb, extract_board_by_contour
from tileDetection import is_tile_present
from DNN_shape_classifier import get_shape_model
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
import cv2

def warp_template(template_img): 
    """
    Warps the template image to a standard size and saves it.
    Args:
        template_img (str): Path to the template image.
    Returns:
        warped_template (numpy.ndarray): Warped template image.
    """
    # warp template_img and show result
    # warped_template = cv2.imread(template_img[img_num-1])
    warped_template = cv2.imread(template_img)
    _, debug_lines, warped_template = detect_board_and_warp(warped_template, output_size=(800, 800))
    # plt.imshow(cv2.cvtColor(warped_template, cv2.COLOR_BGR2RGB))
    
    return warped_template
def align_board(wrp_tmp_path, query_img, output_size=(800, 800), show_details=False , use_sift=True, homography_threshold=0.85):
    """
    Aligns the query image to the template image using SIFT or ORB feature matching.
    Args:
        wrp_tmp_path (str): Path to the warped template image.
        query_img (str): Path to the query image.
        output_size (tuple): Desired output size for the aligned image.
        show_details (bool): Whether to show details of the alignment process.
        use_sift (bool): Whether to use SIFT for feature matching.
        homography_threshold (float): Threshold for homography filtering.
    Returns:
        aligned_img (numpy.ndarray): Aligned image.
    """
    aligned_img = None
    try:
        # Attempt to extract the board using contours
        aligned_img = extract_board_by_contour(query_img, output_size, debug=True)
    except Exception as e:
        print(f"Contour extraction failed: {e}")
        try:
            tpl, qry, aligned_img, image_matches = align_board_sift_orb(wrp_tmp_path, query_img, output_size=(800,800), show_details=show_details , use_sift=use_sift, homography_threshold=homography_threshold)
        except Exception as e:
            print(F"Sift alignment failed: {e}")
    # Read template and query images
    template_img_array = wrp_tmp_path  # Read the template image

    query_img_array = cv2.imread(query_img)  # Read the query image
    if query_img_array is None:
        raise FileNotFoundError(f"Query image not found at path: {query_img}")
    
    return aligned_img

def split_board(warped, padding_cell=0, padding_detection=-0.15):
    """
    Splits the warped board into cells and detection grid.
    Args:
        warped (numpy.ndarray): Warped image of the board.
        padding_cell (int): Padding for cell splitting.
        padding_detection (float): Padding for detection grid splitting.
    Returns:
        detection_grid (dict): Dictionary of detection grid cells.
        cells_grid (dict): Dictionary of cell images.
    """
    detection_grid = split_board_into_cells(warped, 16, padding=padding_detection)
    cells_grid = split_board_into_cells(warped, 16, padding=padding_cell)
    
    return detection_grid, cells_grid



def tile_detection(cells_grid, detection_grid, tile_vs_back_threshol=60, min_foreground_ratio=0.28):
    """
    Detects tiles in the cells and checks if they are present.
    Args:
        cells_grid (dict): Dictionary of cell images.
        detection_grid (dict): Dictionary of detection grid cells.
        tile_vs_back_threshol (int): Threshold for tile vs background detection.
        min_foreground_ratio (float): Minimum ratio of foreground to background for tile detection.
    Returns:
        tiles (list): List of dictionaries containing tile information.
        tiles_present (list): List of dictionaries containing present tile information.
    """ 
    tiles = []
    tiles_present = []

    for label, cell_img in cells_grid.items():
        # Check if a tile is present in the current cell
        is_present = is_tile_present(cells_grid.get(label), threshold=tile_vs_back_threshol, min_foreground_ratio=min_foreground_ratio)
        
        # Add the tile information to the tiles list
        tiles.append({'label': label, 'isPresent': is_present, 'img': cell_img})
        
        # If a tile is present, add it to the tiles_present list
        if is_present:
            tiles_present.append({'label': label, 'isPresent': is_present, 'img': cell_img, 'detection_img': detection_grid.get(label)})
            
    return tiles, tiles_present

def classify_tiles(tiles_present):
    """
    Classifies the tiles using a resnet model.
    Args:
        tiles_present (list): List of dictionaries containing present tile information.
    Returns:
        tiles_classified (list): List of dictionaries containing classified tile information.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = get_shape_model(num_classes=10)
    classifier.load_state_dict(torch.load("resnet_shape.pt", map_location=device))
    classifier.to(device)
    classifier.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    shape_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

    tiles_classified = []

    for tile in tiles_present:
        if not tile['isPresent']:
            continue

        label = tile['label']
        patch_img = tile['img']

        if patch_img is None or not isinstance(patch_img, np.ndarray):
            print(f"[WARN] Invalid image at {label}")
            continue

        try:
            # Convert OpenCV image to PIL
            img_pil = Image.fromarray(cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"[ERROR] Failed to convert image at {label}: {e}")
            continue

        # Apply transform and predict
        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = classifier(input_tensor)
            pred_idx = output.argmax(dim=1).item()
            shape = shape_names[pred_idx]

        tiles_classified.append({'label': label, 'shape': shape, 'img': patch_img})
    
    return tiles_classified
    
def map_color_to_name(color_bgr):
    # Simplified color label matching
    color_bgr = np.array(color_bgr)
    colors = {
        "R":     np.array([60, 60, 200]),
        "B":     np.array([200, 60, 60]),
        "G":     np.array([60, 200, 60]),
        "Y":     np.array([0, 220, 220]),
        "O":     np.array([0, 140, 255]),
        "W":   np.array([255, 255, 255]),
    }

    min_dist = float('inf')
    best_match = "W"
    for name, ref in colors.items():
        dist = np.linalg.norm(color_bgr - ref)
        if dist < min_dist:
            min_dist = dist
            best_match = name
    return best_match

def extract_patch_color(image, row, col, grid_size=(6, 6), min_nonzero_pixels=30):
    h, w = image.shape[:2]
    patch_h, patch_w = h // grid_size[1], w // grid_size[0]

    x1 = col * patch_w
    y1 = row * patch_h
    patch = image[y1:y1 + patch_h, x1:x1 + patch_w]

    # Convert to HSV and apply mask to filter out background-like areas
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))  # adjust as needed

    # Extract only valid pixels
    pixels = patch[mask > 0].reshape(-1, 3)

    if len(pixels) < min_nonzero_pixels:
        return "W"

    # KMeans to get dominant color
    kmeans = KMeans(n_clusters=1, n_init='auto').fit(pixels)
    color = kmeans.cluster_centers_[0].astype(int)

    return map_color_to_name(color)

def classify_tiles_full(tiles_present, tiles_classified):
    """
    Classifies the tiles using a pre-trained model and detects their colors.
    Args:
        tiles_present (list): List of dictionaries containing present tile information.
        tiles_classified (list): List of dictionaries containing classified tile information.
    Returns:
        tiles_full_classified (list): List of dictionaries containing fully classified tile information.
    """
    tiles_full_classified = []

    for tile in tiles_present:
        if not tile['isPresent']:
            continue

        img = tile['img']
        shape = next((t['shape'] for t in tiles_classified if t['label'] == tile['label']), None)
        if shape is None:
            print(f"[WARN] Shape not found for tile {tile['label']}")
            continue

        try:
            # Detect color using extract_patch_color assuming 1 tile = 1 patch
            color = extract_patch_color(img, row=0, col=0, grid_size=(1, 1))  # full image = 1 patch
        except Exception as e:
            print(f"[ERROR] Failed color detection for tile {tile['label']}: {e}")
            color = "unknown"

        # print(f"{tile['label']}: {color}")
        tiles_full_classified.append({'label': tile['label'], 'shape': shape, 'color': color, 'img': img})
    
    return tiles_full_classified