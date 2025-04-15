import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image

def load_sample_materials():
    """
    Load sample material data from JSON file.
    
    Returns:
        dict: Dictionary of sample materials and their properties
    """
    try:
        # Load from file if exists
        if os.path.exists("data/sample_materials.json"):
            with open("data/sample_materials.json", "r") as f:
                return json.load(f)
        else:
            # Return built-in samples if file doesn't exist
            return {
                "Carbon Steel - Normalized": {
                    "material_type": "Carbon Steel",
                    "confidence": 0.93,
                    "properties": {
                        "Hardness (HRC)": 25.3,
                        "Grain Size (μm)": 22.5,
                        "Carbon Content (%)": 0.45,
                        "Ferrite Content (%)": 68.2,
                        "Pearlite Content (%)": 31.8,
                        "Estimated Strength (MPa)": 580
                    },
                    "has_defects": False,
                    "defect_types": [],
                    "defect_locations": [],
                    "defect_confidence": 0.0
                },
                "Stainless Steel - 304": {
                    "material_type": "Stainless Steel",
                    "confidence": 0.89,
                    "properties": {
                        "Hardness (HRC)": 18.7,
                        "Grain Size (μm)": 35.2,
                        "Carbon Content (%)": 0.08,
                        "Chromium Content (%)": 18.5,
                        "Nickel Content (%)": 10.2,
                        "Estimated Strength (MPa)": 650
                    },
                    "has_defects": True,
                    "defect_types": ["inclusion", "grain_boundary_issue"],
                    "defect_locations": [(120, 150, 40, 40), (250, 300, 60, 30)],
                    "defect_confidence": 0.86
                },
                "Aluminum Alloy - 6061": {
                    "material_type": "Aluminum 6061",
                    "confidence": 0.95,
                    "properties": {
                        "Hardness (HB)": 95.5,
                        "Grain Size (μm)": 45.8,
                        "Mg Content (%)": 1.0,
                        "Si Content (%)": 0.65,
                        "Estimated Strength (MPa)": 310
                    },
                    "has_defects": False,
                    "defect_types": [],
                    "defect_locations": [],
                    "defect_confidence": 0.0
                },
                "Ceramic - Alumina": {
                    "material_type": "Ceramic - Alumina",
                    "confidence": 0.92,
                    "properties": {
                        "Density (g/cm³)": 3.85,
                        "Porosity (%)": 1.8,
                        "Grain Size (μm)": 3.2,
                        "Estimated Strength (MPa)": 420
                    },
                    "has_defects": True,
                    "defect_types": ["porosity", "crack"],
                    "defect_locations": [(180, 220, 50, 20), (320, 180, 30, 70)],
                    "defect_confidence": 0.78
                },
                "Metal Matrix Composite": {
                    "material_type": "Metal Matrix Composite",
                    "confidence": 0.84,
                    "properties": {
                        "Fiber Content (%)": 42.5,
                        "Density (g/cm³)": 2.23,
                        "Estimated Strength (MPa)": 780
                    },
                    "has_defects": True,
                    "defect_types": ["crack"],
                    "defect_locations": [(220, 280, 40, 30)],
                    "defect_confidence": 0.82
                }
            }
    except Exception as e:
        print(f"Error loading sample materials: {e}")
        return {}

def process_results(material_results, defect_results):
    """
    Process analysis results into a formatted string for display.
    
    Args:
        material_results: dict with material classification results
        defect_results: dict with defect detection results
        
    Returns:
        str: Formatted string with analysis results
    """
    result_str = f"# Material Analysis Results\n\n"
    
    # Material section
    result_str += f"## Material Type: {material_results['material_type']}\n"
    result_str += f"Confidence: {material_results['confidence']:.2f}\n\n"
    
    # Properties section
    result_str += "### Properties\n"
    for prop, value in material_results['properties'].items():
        result_str += f"- {prop}: {value}\n"
    
    # Defect section
    result_str += "\n## Defect Analysis\n"
    
    if defect_results['has_defects']:
        result_str += f"**Defects Detected** (Confidence: {defect_results['defect_confidence']:.2f})\n\n"
        
        # List detected defect types
        result_str += "### Defect Types\n"
        for defect in defect_results['defect_types']:
            result_str += f"- {defect}\n"
    else:
        result_str += "**No Defects Detected**\n"
    
    return result_str

def generate_heatmap(image):
    """
    Generate a heatmap visualization highlighting regions of interest.
    
    Args:
        image: numpy array (H, W, C) representing the image
        
    Returns:
        heatmap_image: numpy array representing the heatmap visualization
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Apply some image processing to generate a heatmap
        # This is a simplified approach - in a real application, this would
        # be based on the model's attention or feature activation maps
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-1
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Create a colormap
        cmap = plt.cm.jet
        
        # Apply colormap
        heatmap = cmap(magnitude)
        
        # Convert to uint8
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        
        # Convert original image to uint8 RGB if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure heatmap and image have the same dimensions
        if heatmap.shape != image.shape:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            
        # Blend original image with heatmap
        alpha = 0.6
        blended = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        
        return blended
        
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return image

def get_material_info(material_type):
    """
    Get additional information about a material type.
    
    Args:
        material_type: str representing the material type
        
    Returns:
        str: Formatted string with material information
    """
    info = ""
    
    if "Carbon Steel" in material_type:
        info = """
        **Carbon Steel** contains carbon as the main alloying element. The carbon content typically ranges from 0.05% to 2.1% by weight.
        
        **Properties**:
        - Good tensile strength and ductility
        - Easily welded and machined
        - Susceptible to corrosion
        
        **Common Uses**:
        - Automotive parts
        - Construction materials
        - Tools and machinery
        
        **Microstructure Features**:
        - Ferrite (white regions)
        - Pearlite (dark lamellar regions)
        - Possible presence of cementite at grain boundaries
        """
    
    elif "Stainless Steel" in material_type:
        info = """
        **Stainless Steel** contains at least 10.5% chromium, which forms a passive film that protects against corrosion.
        
        **Properties**:
        - Excellent corrosion resistance
        - High strength and ductility
        - Good high-temperature properties
        
        **Common Uses**:
        - Food processing equipment
        - Medical instruments
        - Architecture and construction
        
        **Microstructure Features**:
        - Austenite (in 300-series)
        - Ferrite (in 400-series)
        - Possible martensite in hardened grades
        """
    
    elif "Aluminum" in material_type:
        info = """
        **Aluminum Alloys** contain aluminum as the predominant metal, with added elements like copper, magnesium, and zinc.
        
        **Properties**:
        - Low density (light weight)
        - Good corrosion resistance
        - Excellent thermal and electrical conductivity
        
        **Common Uses**:
        - Aerospace components
        - Automotive parts
        - Construction materials
        
        **Microstructure Features**:
        - Alpha aluminum matrix
        - Intermetallic precipitates
        - Possible dispersoids for strength
        """
    
    elif "Ceramic" in material_type:
        info = """
        **Ceramics** are inorganic, non-metallic materials typically formed through high-temperature processing.
        
        **Properties**:
        - High hardness and wear resistance
        - Low thermal and electrical conductivity
        - Brittle behavior
        
        **Common Uses**:
        - Cutting tools
        - Electrical insulators
        - High-temperature components
        
        **Microstructure Features**:
        - Crystalline grains
        - Possible glassy phase at boundaries
        - Pores or voids (potential defect sites)
        """
    
    elif "Composite" in material_type:
        info = """
        **Composites** consist of two or more materials with significantly different physical or chemical properties.
        
        **Properties**:
        - Tailored strength and stiffness
        - Good fatigue resistance
        - Typically lightweight
        
        **Common Uses**:
        - Aerospace structures
        - Sporting goods
        - Automotive components
        
        **Microstructure Features**:
        - Matrix phase
        - Reinforcement phase (fibers, particles)
        - Interface regions between components
        """
    
    else:
        info = """
        **Material Type Information**
        
        Detailed information for this material type is not available.
        
        The microstructure of materials often reveals important details about:
        - Grain structure and size
        - Phase distribution
        - Processing history
        - Potential defect locations
        """
    
    return info
