import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
import itertools
import os
from typing import Tuple, List, Optional


class Config:
    OUTPUT_FOLDER = './output_results'
    CLAHE_CLIP = 6.0
    CLAHE_GRID = (8, 8)
    GAUSSIAN_KERNEL = (5, 5) 
    BILATERAL_D = 9
    BILATERAL_SIGMA_COLOR = 75
    BILATERAL_SIGMA_SPACE = 75
    MIN_WORM_AREA = 500
    GLUE_KERNEL_SIZE = (3, 3)
    SKELETON_SMOOTHING_S = 50  



def load_image(path: str) -> Optional[np.ndarray]:
    """Loads image and converts to grayscale."""
    img = cv2.imread(path)
    if img is None:
        return None
    return img

def enhance_image(gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Applies CLAHE and Gaussian Blur + Bilateral Filter."""
    clahe = cv2.createCLAHE(clipLimit=Config.CLAHE_CLIP, tileGridSize=Config.CLAHE_GRID)
    enhanced = clahe.apply(gray_img)

    blurred = cv2.GaussianBlur(enhanced, Config.GAUSSIAN_KERNEL, 0)
    filtered = cv2.bilateralFilter(blurred, 
                                   Config.BILATERAL_D, 
                                   Config.BILATERAL_SIGMA_COLOR, 
                                   Config.BILATERAL_SIGMA_SPACE)
    return blurred, filtered

def segment_worm(filtered_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Thresholds, filters contours, glues parts, and keeps the largest object."""

    ret, binary = cv2.threshold(filtered_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_mask = np.zeros_like(binary)
    debug_mask = np.zeros_like(binary)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)

        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        short = min(w, h)
        
        if area > Config.MIN_WORM_AREA and short > 0:
            cv2.drawContours(candidate_mask, [cnt], -1, 255, -1) 
            cv2.drawContours(debug_mask, [cnt], -1, 255, -1)     
        else:
            cv2.drawContours(debug_mask, [cnt], -1, 100, -1)     

    glue_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Config.GLUE_KERNEL_SIZE)
    glued_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, glue_kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(glued_mask, connectivity=8)
    final_mask = np.zeros_like(binary)
    
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask[labels == largest_label] = 255
        
    return final_mask, debug_mask

def find_centerline(binary_mask: np.ndarray) -> Tuple[float, List, Tuple[List, List]]:
    """Skeletonizes the mask and finds the longest path (centerline)."""

    skeleton = skeletonize(binary_mask > 0)
    length_px, path = _get_longest_graph_path(skeleton)
    
    smooth_x, smooth_y = [], []
    if len(path) > 10:
        try:
            px, py = zip(*path)
            tck, u = splprep([px, py], s=Config.SKELETON_SMOOTHING_S) 
            u_new = np.linspace(0, 1, len(path))
            smooth_x, smooth_y = splev(u_new, tck)
        except Exception:
            pass 
            
    return length_px, path, (smooth_x, smooth_y)

def _get_longest_graph_path(skeleton_img):
    """Internal helper: Converts skeleton pixels to graph and finds longest path."""
    y_indices, x_indices = np.nonzero(skeleton_img)
    points = list(zip(x_indices, y_indices))
    
    if len(points) < 2: return 0.0, []

    G = nx.Graph()
    for x, y in points:
        G.add_node((x, y))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx_idx, ny_idx = x + dx, y + dy
                if (nx_idx, ny_idx) in points:
                    dist = np.sqrt(dx**2 + dy**2)
                    G.add_edge((x, y), (nx_idx, ny_idx), weight=dist)

    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    search_nodes = endpoints if len(endpoints) >= 2 else list(G.nodes())
    
    max_len = 0.0
    best_path = []
    
    for start, end in itertools.combinations(search_nodes, 2):
        try:
            length = nx.shortest_path_length(G, start, end, weight='weight')
            if length > max_len:
                max_len = length
                best_path = nx.shortest_path(G, start, end)
        except nx.NetworkXNoPath:
            continue
            
    return max_len, best_path

def visualize_results(img_orig, gaussian_preview, final_mask, debug_mask, centerline_data, length_px, save_path):
    """Plots and saves the analysis results."""
    smooth_x, smooth_y = centerline_data
    
    fig, ax = plt.subplots(1, 4, figsize=(18, 6))

    ax[0].imshow(gaussian_preview, cmap='gray')
    ax[0].set_title("1. CLAHE + Gaussian Blur")

    ax[1].imshow(final_mask, cmap='gray')
    ax[1].set_title("2. Final Mask")

    ax[2].imshow(debug_mask, cmap='gray')
    ax[2].set_title("3. Candidates (Gray=Rejected)")

    ax[3].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    if len(smooth_x) > 0:
        ax[3].plot(smooth_x, smooth_y, 'r-', linewidth=3, label='Centerline')
    ax[3].set_title(f"4. Length: {length_px:.1f} px")
    ax[3].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Saved] {save_path}")
    # plt.show() 
    plt.close(fig) 


def process_image(image_path: str):
    """Orchestrates the processing for a single image."""
    print(f"Processing: {image_path}")

    img = load_image(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian_preview, filtered_img = enhance_image(gray)

    final_mask, debug_mask = segment_worm(filtered_img)
    length_px, raw_path, smooth_path = find_centerline(final_mask)

    filename = os.path.basename(image_path)
    save_path = os.path.join(Config.OUTPUT_FOLDER, f"result_{filename}")
    
    visualize_results(img, gaussian_preview, final_mask, debug_mask, smooth_path, length_px, save_path)

if __name__ == "__main__":
    os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
    
    input_folder = './data'

    files_to_process = [
        '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png',
        '16_train.png', '17_train.png', '64_train.png', '64.png', 
        '77_train.png', '80.png'
    ]

    for f in files_to_process:
        full_path = os.path.join(input_folder, f)
        if os.path.exists(full_path):
            process_image(full_path)
        else:
            print(f"[Skip] File not found: {full_path}")