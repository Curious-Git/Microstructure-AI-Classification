"""
Module for loading realistic material microstructure images
"""
import os
import glob
import numpy as np
import random
from PIL import Image

# Directory for reference images
SAMPLES_DIR = "data/microstructure_samples"

def get_material_image(material_type):
    """
    Get a realistic microstructure image for a specific material type
    
    Args:
        material_type: Name of the material type (e.g., "Carbon Steel", "Ceramic")
        
    Returns:
        image: numpy array representing the image, or None if not found
    """
    # Sanitize material type to match filename pattern
    sanitized_type = material_type.lower().replace(" ", "_").replace("-", "_")
    
    # Find all matching image files
    pattern = os.path.join(SAMPLES_DIR, f"{sanitized_type}_*.jpg")
    image_files = glob.glob(pattern)
    
    # For composite materials, try alternative patterns
    if len(image_files) == 0 and "matrix" in material_type.lower():
        pattern = os.path.join(SAMPLES_DIR, "metal_matrix_composite_*.jpg")
        image_files = glob.glob(pattern)
    
    # For aluminum alloys, try the general aluminum pattern
    if len(image_files) == 0 and "aluminum" in material_type.lower():
        pattern = os.path.join(SAMPLES_DIR, "aluminum_alloy_*.jpg")
        image_files = glob.glob(pattern)
    
    if image_files:
        # Choose a random image from the available options
        image_path = random.choice(image_files)
        try:
            # Load and convert to numpy array
            image = np.array(Image.open(image_path))
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    
    # Return None if no image found
    return None

def get_available_materials():
    """
    Get a list of materials that have realistic microstructure images available
    
    Returns:
        materials: list of material types with available images
    """
    if not os.path.exists(SAMPLES_DIR):
        return []
    
    # Get all image files
    image_files = glob.glob(os.path.join(SAMPLES_DIR, "*.jpg"))
    
    # Extract unique material types
    materials = set()
    for image_path in image_files:
        filename = os.path.basename(image_path)
        # Extract material name from filename (e.g., carbon_steel_1.jpg -> Carbon Steel)
        material_name = filename.split("_")[0]
        if material_name == "metal":
            materials.add("Metal Matrix Composite")
        elif material_name == "carbon":
            materials.add("Carbon Steel")
        elif material_name == "stainless":
            materials.add("Stainless Steel")
        elif material_name == "aluminum":
            materials.add("Aluminum Alloy")
        elif material_name == "ceramic":
            materials.add("Ceramic")
    
    return list(materials)