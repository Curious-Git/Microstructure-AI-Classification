import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import random

class MaterialClassifier:
    """
    A deep learning model for classifying material types from microstructure images.
    For this implementation, we'll simulate the model's behavior since we don't have access
    to pre-trained weights specifically for materials science.
    """
    def __init__(self):
        # In a real application, we would load pre-trained weights here
        # For example, using transfer learning with a ResNet model
        
        # Use a ResNet model as the base model
        self.model = models.resnet50(pretrained=True)
        
        # Modify the final layer for our specific task
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 15)  # 15 different material types
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Define material classes
        self.classes = [
            "Carbon Steel", "Stainless Steel", "Alloy Steel", "Tool Steel",
            "Aluminum 6061", "Aluminum 7075", "Aluminum Cast", 
            "Copper Alloy", "Titanium Alloy", "Nickel Alloy",
            "Ceramic - Alumina", "Ceramic - Zirconia", "Ceramic - Silicon Carbide",
            "Polymer Composite", "Metal Matrix Composite"
        ]
        
        # Set model to evaluation mode
        self.model.eval()
        
        print("Material classifier model loaded")
    
    def __call__(self, image):
        """
        Classify a material from its microstructure image.
        
        Args:
            image: numpy array (H, W, C) representing the image
            
        Returns:
            material_type: string representing the predicted material type
            confidence: float representing the confidence score (0-1)
            properties: dictionary of predicted material properties
        """
        # In a real application, we would process the image through the model
        # Here, we'll simulate the model's prediction
        
        try:
            # Convert to tensor (normalize)
            if len(image.shape) == 2:
                # If grayscale, convert to RGB
                image = np.stack([image, image, image], axis=2)
            
            # Ensure image is uint8 with 3 channels
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
                
            if image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Remove alpha channel
                
            # For demo purposes, we'll use simple image statistics to determine the material type
            # In a real application, this would be the output of the deep learning model
            gray_img = np.mean(image, axis=2)
            avg_intensity = np.mean(gray_img)
            std_intensity = np.std(gray_img)
            
            # Use image stats to simulate different material types
            # Adjust classification thresholds to better identify ceramics
            # Check for ceramic characteristics first (high contrast, specific color patterns)
            if (avg_intensity > 180 and avg_intensity < 240 and std_intensity > 40) or \
               (np.mean(image[:,:,0]) > np.mean(image[:,:,2]) + 20):  # Reddish tint common in ceramics
                # Select a specific ceramic type
                ceramic_types = [10, 11, 12]  # Indices for ceramic materials
                material_idx = random.choice(ceramic_types)
            # Then check for other materials
            elif avg_intensity > 200:
                material_idx = 4  # Aluminum
            elif avg_intensity > 150:
                material_idx = 0  # Carbon Steel
            elif avg_intensity > 100:
                material_idx = 1  # Stainless Steel
            elif std_intensity > 50:
                material_idx = 9  # Nickel Alloy
            else:
                material_idx = 13  # Composite
                
            # Add some randomness to make it more realistic
            r = random.random()
            if r > 0.8:
                material_idx = (material_idx + 1) % len(self.classes)
                
            material_type = self.classes[material_idx]
            
            # Generate a realistic confidence score
            confidence = 0.7 + (random.random() * 0.25)
            
            # Generate some material properties based on the material type
            properties = self._generate_material_properties(material_type)
            
            return material_type, confidence, properties
            
        except Exception as e:
            print(f"Error classifying material: {e}")
            return "Unknown", 0.0, {}
    
    def _generate_material_properties(self, material_type):
        """Generate simulated material properties based on material type."""
        properties = {}
        
        if "Steel" in material_type:
            properties["Hardness (HRC)"] = round(random.uniform(20, 65), 1)
            properties["Grain Size (μm)"] = round(random.uniform(5, 50), 1)
            properties["Carbon Content (%)"] = round(random.uniform(0.1, 1.5), 2)
            
            if "Carbon" in material_type:
                properties["Ferrite Content (%)"] = round(random.uniform(60, 95), 1)
                properties["Pearlite Content (%)"] = round(100 - properties["Ferrite Content (%)"], 1)
            elif "Stainless" in material_type:
                properties["Chromium Content (%)"] = round(random.uniform(16, 25), 1)
                properties["Nickel Content (%)"] = round(random.uniform(8, 20), 1)
            elif "Alloy" in material_type:
                properties["Alloy Elements (%)"] = round(random.uniform(1, 5), 1)
                
        elif "Aluminum" in material_type:
            properties["Hardness (HB)"] = round(random.uniform(30, 150), 1)
            properties["Grain Size (μm)"] = round(random.uniform(20, 100), 1)
            
            if "6061" in material_type:
                properties["Mg Content (%)"] = round(random.uniform(0.8, 1.2), 2)
                properties["Si Content (%)"] = round(random.uniform(0.4, 0.8), 2)
            elif "7075" in material_type:
                properties["Zn Content (%)"] = round(random.uniform(5.1, 6.1), 2)
                properties["Mg Content (%)"] = round(random.uniform(2.1, 2.9), 2)
                
        elif "Ceramic" in material_type:
            properties["Density (g/cm³)"] = round(random.uniform(2.5, 6.0), 2)
            properties["Porosity (%)"] = round(random.uniform(0, 5), 1)
            properties["Grain Size (μm)"] = round(random.uniform(1, 10), 1)
            
        elif "Composite" in material_type:
            properties["Fiber Content (%)"] = round(random.uniform(30, 70), 1)
            properties["Density (g/cm³)"] = round(random.uniform(1.5, 2.5), 2)
            
        # Add some common properties for all materials
        properties["Estimated Strength (MPa)"] = round(random.uniform(200, 1200), 0)
            
        return properties


class DefectDetector:
    """
    A deep learning model for detecting defects in material microstructure images.
    For this implementation, we'll simulate the model's behavior.
    """
    def __init__(self):
        # In a real application, we would load pre-trained weights here
        # For example, using a Faster R-CNN or YOLO model
        
        # Use a pretrained Faster R-CNN model
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Modify for our specific classes of defects
        num_classes = 5  # 4 defect types + background
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
        # Define defect classes
        self.classes = ["background", "crack", "porosity", "inclusion", "grain_boundary_issue"]
        
        # Set model to evaluation mode
        self.model.eval()
        
        print("Defect detector model loaded")
    
    def __call__(self, image):
        """
        Detect defects in a material microstructure image.
        
        Args:
            image: numpy array (H, W, C) representing the image
            
        Returns:
            has_defects: boolean indicating whether defects were detected
            defect_types: list of strings representing the detected defect types
            defect_locations: list of tuples (x, y, w, h) representing bounding boxes
            defect_confidence: float representing the overall confidence score (0-1)
        """
        # In a real application, we would process the image through the model
        # Here, we'll simulate the model's prediction
        
        try:
            # For demo purposes, we'll use simple image statistics to simulate defect detection
            gray_img = np.mean(image, axis=2) if len(image.shape) == 3 else image
            
            # Compute local standard deviation (a simple way to find potential defects)
            from scipy.ndimage import uniform_filter, standard_deviation
            
            # Simulate defect detection based on image statistics
            std_dev = np.std(gray_img)
            mean_val = np.mean(gray_img)
            
            # Decide if the image has defects (for simulation)
            # Images with high standard deviation or very low/high mean values 
            # are more likely to have defects for our simulation
            has_defects = (std_dev > 40) or (mean_val < 50) or (mean_val > 200) or (random.random() < 0.3)
            
            defect_types = []
            defect_locations = []
            
            if has_defects:
                # Decide how many defects to simulate
                num_defects = random.randint(1, 4)
                
                # Generate random defect types and locations
                available_defects = ["crack", "porosity", "inclusion", "grain_boundary_issue"]
                
                for _ in range(num_defects):
                    # Choose a random defect type
                    defect_type = random.choice(available_defects)
                    if defect_type not in defect_types:
                        defect_types.append(defect_type)
                    
                    # Generate a random bounding box
                    height, width = image.shape[:2]
                    x = random.randint(0, width - 50)
                    y = random.randint(0, height - 50)
                    w = random.randint(30, min(100, width - x))
                    h = random.randint(30, min(100, height - y))
                    
                    defect_locations.append((x, y, w, h))
            
            # Generate a realistic confidence score
            defect_confidence = 0.65 + (random.random() * 0.3) if has_defects else 0.0
            
            return has_defects, defect_types, defect_locations, defect_confidence
            
        except Exception as e:
            print(f"Error detecting defects: {e}")
            return False, [], [], 0.0
