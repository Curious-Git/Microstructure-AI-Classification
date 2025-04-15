"""
Module for managing reference material images and datasets
"""
import os
import json
import numpy as np
from PIL import Image
import cv2
import glob

# Directory for reference images
SAMPLES_DIR = "data/microstructure_samples"

def get_available_reference_materials():
    """
    Get a list of available reference materials with actual microstructure images
    
    Returns:
        materials_dict: Dictionary mapping material names to image paths
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    materials_dict = {}
    
    # Get all image files
    image_files = glob.glob(os.path.join(SAMPLES_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(SAMPLES_DIR, "*.png"))
    
    # Process file names to extract material types
    for image_path in image_files:
        filename = os.path.basename(image_path)
        # Extract material name from filename (e.g., carbon_steel_1.jpg -> Carbon Steel)
        material_name = os.path.splitext(filename)[0]  # Remove extension
        material_name = material_name.split("_")[:-1]  # Remove the numbering
        material_name = " ".join(material_name).title().replace("_", " ")
        
        # Some special case handling
        if "Mmc" in material_name:
            material_name = "Metal Matrix Composite"
        
        # Add to dictionary (possibly overwriting previous entries)
        materials_dict[material_name] = image_path
    
    # If no materials found, add default reference to standard dataset
    if not materials_dict:
        materials_dict = {
            "Carbon Steel": "No reference image available",
            "Stainless Steel": "No reference image available",
            "Aluminum Alloy": "No reference image available",
            "Ceramic": "No reference image available",
            "Metal Matrix Composite": "No reference image available"
        }
    
    return materials_dict

def load_reference_image(material_name):
    """
    Load a reference image for a specific material
    
    Args:
        material_name: Name of the material
        
    Returns:
        image: numpy array representing the image, or None if not found
    """
    reference_materials = get_available_reference_materials()
    
    if material_name in reference_materials:
        image_path = reference_materials[material_name]
        
        if os.path.isfile(image_path):
            try:
                # Load and convert to numpy array
                image = np.array(Image.open(image_path))
                return image
            except Exception as e:
                print(f"Error loading reference image for {material_name}: {e}")
    
    return None

def download_missing_references():
    """
    Check if reference materials exist, and if not, run the download script
    
    Returns:
        success: boolean indicating whether references are available
    """
    # Check if we have any reference images
    if not os.path.exists(SAMPLES_DIR) or not os.listdir(SAMPLES_DIR):
        try:
            print("Downloading reference material images...")
            import download_reference_images
            return True
        except Exception as e:
            print(f"Error downloading reference images: {e}")
            return False
    
    return True

# Make sure we have reference materials
download_missing_references()