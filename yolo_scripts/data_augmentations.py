import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from pathlib import Path

def apply_clahe(image, clip_limit, tile_grid_size):
    """Apply CLAHE with given parameters"""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    transform = A.Compose([
        A.CLAHE(
            clip_limit=clip_limit,
            tile_grid_size=(tile_grid_size, tile_grid_size),
            p=1.0
        )
    ])
    result = transform(image=image)['image']
    return result

def visualize_clahe_combinations(image_path):
    """
    Create a grid of images with different CLAHE parameters
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Parameters to test
    clip_limits = [1.0, 2.0, 4.0]
    grid_sizes = [4, 8, 16]
    
    # Create 3x3 grid (9 subplots)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.suptitle('CLAHE Parameter Combinations', fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.ravel()
    
    # Apply different combinations
    idx = 0
    for clip in clip_limits:
        for grid in grid_sizes:
            enhanced = apply_clahe(image, clip, grid)
            axes[idx].imshow(enhanced)
            axes[idx].set_title(f'Clip: {clip}, Grid: {grid}x{grid}')
            axes[idx].axis('off')
            idx += 1
    
    plt.tight_layout()
    
    # Show original image separately
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.show()

# Example usage
try:
    image_path = Path("M:/Unused/TauCellDL/images/train/747297/747297 [d=0.98892,x=25633,y=17563,w=506,h=507].png")
    visualize_clahe_combinations(image_path)
except Exception as e:
    print(f"Error: {e}")