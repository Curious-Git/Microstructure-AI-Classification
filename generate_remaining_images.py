"""
Generate realistic microstructure images for remaining materials
"""
import os
from download_reference_images import create_realistic_microstructure_image
from PIL import Image

# Materials to generate images for
REMAINING_MATERIALS = [
    "Carbon Steel",
    "Stainless Steel",
    "Aluminum Alloy"
]

# Directory for reference images
SAMPLES_DIR = "data/microstructure_samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

def generate_missing_materials():
    """Generate the missing material images with realistic microstructures"""
    for material_name in REMAINING_MATERIALS:
        sanitized_name = material_name.replace(" ", "_").lower()
        
        # Remove any placeholder images
        placeholder_path = os.path.join(SAMPLES_DIR, f"{sanitized_name}_placeholder.jpg")
        if os.path.exists(placeholder_path):
            os.remove(placeholder_path)
            print(f"Removed placeholder for {material_name}")
        
        print(f"Creating realistic microstructure for {material_name}")
        
        # Create multiple images per material type for variety
        for i in range(3):
            filename = f"{sanitized_name}_{i+1}.jpg"
            target_path = os.path.join(SAMPLES_DIR, filename)
            
            # Create the realistic microstructure image
            img_array = create_realistic_microstructure_image(material_name)
            
            # Save the image
            Image.fromarray(img_array).save(target_path)
            print(f"  Created realistic image: {filename}")
            
        print(f"  Created 3 images for {material_name}")

if __name__ == "__main__":
    print("Generating realistic microstructure images for remaining materials...")
    generate_missing_materials()
    print("\nDone! All reference materials are now available.")