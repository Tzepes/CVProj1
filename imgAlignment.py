import cv2
import numpy as np
from typing import List, Tuple
from scipy.spatial import distance as dist
from utilities import ShowImage, ShowKeypoints

def get_keypoints_and_features_SIFT(image, show_keypoints=False) -> tuple:  
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    sift = cv2.SIFT_create() 
    keypoints = sift.detect(gray_image, None)
    keypoints, features = sift.compute(gray_image, keypoints) 
    
    # if show_keypoints:
    #     ShowKeypoints(image, keypoints, title="Keypoints SIFT")
    #     ShowImage(image, "Keypoints SIFT")
        
    return keypoints, features

def get_keypoints_and_descriptors_ORB(image, show_details=False) -> tuple:
    MAX_NUM_FEATURES = 500
    ORB = cv2.ORB_create(MAX_NUM_FEATURES)
    keypoints, descriptors = ORB.detectAndCompute(image, None)
    if show_details:
        print(f"[INFO] Number of keypoints: {len(keypoints)}")
        print(f"[INFO] Keypoints: {keypoints}")
        print(f"[INFO] Descriptors: {descriptors}")
    
    return keypoints, descriptors


def match_features(features_source, features_dest) -> list[list[cv2.DMatch]]:
    """
    Match features from the source image with the features from the destination image.
    :return: list[list[cv2.DMatch]] - The result of the matching. For each set of features from the source image,
    it returns the first 'K' matchings from the destination images.
    """
    # Convert descriptors to float32 if they are not already
    if features_source.dtype != np.float32:
        features_source = np.float32(features_source)
    if features_dest.dtype != np.float32:
        features_dest = np.float32(features_dest)
 
    feature_matcher = cv2.BFMatcher()
    matches = feature_matcher.knnMatch(features_source, features_dest, k=2)   
    return matches


def generate_homography(all_matches: list[cv2.DMatch], keypoints_source: list[cv2.KeyPoint], keypoints_dest: list[cv2.KeyPoint],
                        ratio: float = 0.6, ransac_rep: float = 3.0, max_matches: int = 20):
    """
    :param all_matches [DMatch]
    :param keypoints_source [cv.Point]
    :param ratio - Lowe's ratio test (the ratio 1st neighbour distance / 2nd neighbour distance)
    :param keypoints_source: nd.array [Nx2] (x, y coordinates)
    :param keypoints_dest: nd.array [Nx2] (x, y coordinates)
    :param ransac_rep: float. The threshold in the RANSAC algorithm.
    :return: The homography matrix.
    
    class DMatch:
        distance - Distance between descriptors. The lower, the better it is.
        imgIdx - Index of the train image
        queryIdx - Index of the descriptor in query descriptors
        trainIdx - Index of the descriptor in train descriptors
    
    class KeyPoint:
        pt - The x, y coordinates of a point.
    """
    if not all_matches:
        return None

    # Apply Lowe's ratio test
    good_matches = []
    for match in all_matches:
        if len(match) == 2 and (match[0].distance / match[1].distance) < ratio:
            good_matches.append(match[0])

    print(f"[INFO] Good matches before filtering: {len(good_matches)}")

    if len(good_matches) < 10:
        print(f"[WARN] Not enough good matches: {len(good_matches)}")
        return None

    # Prepare points
    points_source = np.float32([keypoints_source[m.queryIdx].pt for m in good_matches])
    points_dest = np.float32([keypoints_dest[m.trainIdx].pt for m in good_matches])

    H, status = cv2.findHomography(points_source, points_dest, cv2.RANSAC, ransac_rep)

    return H


def align_board_sift_orb(template_img: np.array, query_image_path: str, output_size=(800,800), show_details=False, use_sift=False, homography_threshold=0.75) -> tuple:
    """
    Align the query (skewed) board image to match the template (perfect board).
    """
    
    template = template_img
    query = cv2.imread(query_image_path)
    
    template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    query = cv2.cvtColor(query, cv2.COLOR_BGR2HSV)
    
    if use_sift:
        keypoints_template, features_template = get_keypoints_and_features_SIFT(template, show_keypoints=show_details)
        keypoints_query, features_query = get_keypoints_and_features_SIFT(query, show_keypoints=show_details)
    else:
        keypoints_template, features_template = get_keypoints_and_descriptors_ORB(template, show_details=show_details)
        keypoints_query, features_query = get_keypoints_and_descriptors_ORB(query, show_details=show_details)    

    all_matches = match_features(features_template, features_query)

    # if show_details:
    #     matches = sorted(all_matches, key=lambda x: x[0].distance)
    #     matches = matches[:20]
    image_matches = cv2.drawMatchesKnn(template, keypoints_template,
                                        query, keypoints_query,
                                        all_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # ShowImage(image_matches, 'matches')


    # # Notice the order: source = template, destination = query
    H = generate_homography(all_matches, keypoints_template, keypoints_query, ratio=homography_threshold, ransac_rep=4.0)
    H = np.linalg.inv(H)  # Invert the homography matrix to warp the query image to the template

    if H is None:
        raise ValueError("Homography could not be computed.")

    shape = (template.shape[1], template.shape[0])
    result = cv2.warpPerspective(query, H, shape)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    return template, query, result, image_matches



def extract_board_by_contour(image_path, output_size=(800, 800), debug=False):
    warped = None
    image = cv2.imread(image_path)
    orig = image.copy()
    img_area = image.shape[0] * image.shape[1]
    
    # Convert to grayscale and blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=11, sigmaColor=17, sigmaSpace=17)

    # Canny edge detection
    edged = cv2.Canny(gray, 1, 10, apertureSize=3)
    edged = cv2.dilate(edged, None, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    
    #show contours
    debug_img = image.copy()
    for i, contour in enumerate(contours):
        cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 3)  # Draw each contour in green

    # Plot the image with the contours

    board_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.1 * img_area:  # skip tiny contours
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            # Check if it's rectangular-ish
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.8 < aspect_ratio < 1.2:  # roughly square/rectangular
                board_contour = approx
                break

    if board_contour is None:
        raise RuntimeError("Could not find a valid board contour.")
    
    # Draw the detected contour
    debug_img = cv2.drawContours(orig.copy(), [board_contour], -1, (0,255,0), 5)

    # Order points
    def order_points(pts):
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(board_contour)
    dst = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")

    # Warp the image
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, output_size)

    return warped