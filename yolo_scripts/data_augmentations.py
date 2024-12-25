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

def visualize_clahe_combinations(image_dir):
    """
    Create a grid of images with different CLAHE parameters for all images in directory
    """
    # Parameters to test
    clip_limits = [1.0, 2.0, 3.0]
    grid_size = 8  # Fixed grid size
    
    # Get all image files
    image_files = list(Path(image_dir).glob('*.png'))
    
    for image_path in image_files:
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not load image from {image_path}")
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            plt.suptitle(f'CLAHE Parameters - {image_path.name}', fontsize=16)
            
            # Apply different clip limits
            for idx, clip in enumerate(clip_limits):
                enhanced = apply_clahe(image, clip, grid_size)
                axes[idx].imshow(enhanced)
                axes[idx].set_title(f'Clip: {clip}, Grid: {grid_size}x{grid_size}')
                axes[idx].axis('off')
            
            plt.tight_layout()
            
            # Show original image separately
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.show()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Example usage
try:
    image_dir = "M:/Unused/TauCellDL/images/validation/747352/"
    visualize_clahe_combinations(image_dir)
except Exception as e:
    print(f"Error: {e}")