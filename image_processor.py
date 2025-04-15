import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

def preprocess_image(image):
    """
    Preprocesses an image for deep learning models.
    
    Args:
        image: numpy array (H, W, C) representing the image
        
    Returns:
        preprocessed_image: numpy array ready for model input
    """
    try:
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Remove alpha channel
            
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Resize to standard input size (224x224 for most models)
        resized = cv2.resize(image, (224, 224))
        
        # Convert to float32 and normalize to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        # Return a blank image if preprocessing fails
        return np.zeros((224, 224, 3), dtype=np.float32)

def apply_contrast(image, contrast_factor=1.0, brightness_value=0):
    """
    Applies contrast and brightness adjustments to an image.
    
    Args:
        image: numpy array (H, W, C) representing the image
        contrast_factor: float representing the contrast adjustment
        brightness_value: int representing the brightness adjustment
        
    Returns:
        adjusted_image: numpy array with adjusted contrast and brightness
    """
    try:
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Apply contrast
        adjusted = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_value)
        
        return adjusted
    
    except Exception as e:
        print(f"Error applying contrast: {e}")
        return image

def apply_zoom(image, zoom_factor=1.0):
    """
    Applies zoom to an image by cropping the center and resizing.
    
    Args:
        image: numpy array (H, W, C) representing the image
        zoom_factor: float representing the zoom level (1.0 = no zoom)
        
    Returns:
        zoomed_image: numpy array with applied zoom
    """
    try:
        # If zoom factor is 1.0 or less, return original image
        if zoom_factor <= 1.0:
            return image
            
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Calculate new dimensions after zoom
        new_h = int(h / zoom_factor)
        new_w = int(w / zoom_factor)
        
        # Calculate crop starting points
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        
        # Crop the center portion
        cropped = image[start_h:start_h+new_h, start_w:start_w+new_w]
        
        # Resize back to original dimensions
        zoomed = cv2.resize(cropped, (w, h))
        
        return zoomed
    
    except Exception as e:
        print(f"Error applying zoom: {e}")
        return image

def create_image_tensor(image):
    """
    Creates a tensor from an image for deep learning models.
    
    Args:
        image: numpy array (H, W, C) representing the image
        
    Returns:
        tensor: PyTorch tensor ready for model input
    """
    try:
        # Define transformations
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert image to PIL image
        if isinstance(image, np.ndarray):
            # Ensure image is in correct format
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
                
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image, image, image], axis=2)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Remove alpha channel
                
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
            
        # Apply transformations and convert to tensor
        tensor = transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    except Exception as e:
        print(f"Error creating image tensor: {e}")
        # Return a blank tensor if conversion fails
        return torch.zeros((1, 3, 224, 224))
