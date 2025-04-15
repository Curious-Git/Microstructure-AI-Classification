import os
import json
import datetime
import base64
from io import BytesIO
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, LargeBinary, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import numpy as np
from PIL import Image

# Create database connection with error handling
try:
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        print("WARNING: DATABASE_URL environment variable not set")
        DATABASE_URL = "sqlite:///materials.db"  # Fallback to SQLite
    
    # For PostgreSQL, ensure we have the correct driver
    if DATABASE_URL.startswith('postgres'):
        # Make sure to use the psycopg2 driver
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql+psycopg2://')
    
    print(f"Connecting to database with URL: {DATABASE_URL.split('@')[0]}@...")
    engine = create_engine(DATABASE_URL)
    Base = declarative_base()
    Session = sessionmaker(bind=engine)
    
    # Test the connection
    connection = engine.connect()
    connection.close()
    print("Database connection successful")
    db_connection_error = None
except Exception as e:
    print(f"Error creating database connection: {e}")
    db_connection_error = str(e)
    # Create dummy objects to prevent app from crashing
    Base = declarative_base()
    engine = None
    Session = None

class MaterialAnalysis(Base):
    """Table for storing material analysis results"""
    __tablename__ = 'material_analyses'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=func.now())
    image_filename = Column(String(255), nullable=True)
    image_data = Column(LargeBinary, nullable=True)  # Store image binary data
    
    # Material classification results
    material_type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    properties = Column(JSON, nullable=True)  # Store material properties as JSON
    
    # Defect detection results
    has_defects = Column(Boolean, default=False)
    defect_types = Column(JSON, nullable=True)  # Store as JSON array
    defect_locations = Column(JSON, nullable=True)  # Store as JSON array
    defect_confidence = Column(Float, nullable=True)
    
    # Image preprocessing parameters
    contrast = Column(Float, default=1.0)
    brightness = Column(Integer, default=0)
    zoom_factor = Column(Float, default=1.0)
    
    # Additional metadata
    notes = Column(Text, nullable=True)
    user_label = Column(String(255), nullable=True)  # User-provided label/name for the analysis

class CustomMaterial(Base):
    """Table for storing custom material samples added by users"""
    __tablename__ = 'custom_materials'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    material_type = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    properties = Column(JSON, nullable=True)  # Store material properties as JSON
    has_defects = Column(Boolean, default=False)
    defect_types = Column(JSON, nullable=True)  # Store as JSON array
    defect_locations = Column(JSON, nullable=True)  # Store as JSON array
    defect_confidence = Column(Float, nullable=True)
    image_data = Column(LargeBinary, nullable=True)  # Store sample image binary data
    created_at = Column(DateTime, default=func.now())
    notes = Column(Text, nullable=True)

# Create tables
def init_db():
    """Initialize database tables"""
    try:
        if engine is not None:
            Base.metadata.create_all(engine)
            return True
        else:
            print("Cannot initialize database - engine is None")
            return False
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

def save_analysis(material_results, defect_results, image=None, filename=None, preprocessing=None, user_label=None, notes=None):
    """
    Save analysis results to the database
    
    Args:
        material_results: dict with material classification results
        defect_results: dict with defect detection results
        image: numpy array or PIL Image (optional)
        filename: original filename (optional)
        preprocessing: dict with preprocessing parameters (optional)
        user_label: user-provided name/label (optional)
        notes: additional notes (optional)
        
    Returns:
        analysis_id: ID of the saved analysis record
    """
    try:
        session = Session()
        
        # Create new analysis record
        analysis = MaterialAnalysis(
            material_type=material_results['material_type'],
            confidence=material_results['confidence'],
            properties=material_results['properties'],
            has_defects=defect_results['has_defects'],
            defect_types=defect_results['defect_types'],
            defect_locations=defect_results['defect_locations'],
            defect_confidence=defect_results.get('defect_confidence', 0.0),
            user_label=user_label,
            notes=notes
        )
        
        # Add preprocessing params if provided
        if preprocessing:
            analysis.contrast = preprocessing.get('contrast', 1.0)
            analysis.brightness = preprocessing.get('brightness', 0)
            analysis.zoom_factor = preprocessing.get('zoom_factor', 1.0)
            
        # Add filename if provided
        if filename:
            analysis.image_filename = filename
            
        # Add image data if provided
        if image is not None:
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_img = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_img = image
            else:
                pil_img = None
                
            # Convert PIL Image to binary data
            if pil_img:
                img_byte_arr = BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                analysis.image_data = img_byte_arr.getvalue()
        
        # Save to database
        session.add(analysis)
        session.commit()
        analysis_id = analysis.id
        session.close()
        
        return analysis_id
    
    except Exception as e:
        print(f"Error saving analysis to database: {e}")
        if 'session' in locals():
            session.close()
        return None

def get_analysis(analysis_id):
    """
    Retrieve analysis by ID
    
    Args:
        analysis_id: ID of the analysis to retrieve
        
    Returns:
        analysis_data: dict with analysis data
    """
    try:
        session = Session()
        analysis = session.query(MaterialAnalysis).filter_by(id=analysis_id).first()
        
        if not analysis:
            session.close()
            return None
            
        # Prepare image data
        image_data_base64 = None
        if analysis.image_data:
            image_data_base64 = base64.b64encode(analysis.image_data).decode('utf-8')
        
        # Convert to dictionary
        analysis_data = {
            'id': analysis.id,
            'timestamp': analysis.timestamp.isoformat(),
            'image_filename': analysis.image_filename,
            'image_data_base64': image_data_base64,
            'material_results': {
                'material_type': analysis.material_type,
                'confidence': analysis.confidence,
                'properties': analysis.properties
            },
            'defect_results': {
                'has_defects': analysis.has_defects,
                'defect_types': analysis.defect_types,
                'defect_locations': analysis.defect_locations,
                'defect_confidence': analysis.defect_confidence
            },
            'preprocessing': {
                'contrast': analysis.contrast,
                'brightness': analysis.brightness,
                'zoom_factor': analysis.zoom_factor
            },
            'notes': analysis.notes,
            'user_label': analysis.user_label
        }
        
        session.close()
        return analysis_data
    
    except Exception as e:
        print(f"Error retrieving analysis from database: {e}")
        if 'session' in locals():
            session.close()
        return None

def get_recent_analyses(limit=10):
    """
    Get most recent analyses
    
    Args:
        limit: maximum number of records to return
        
    Returns:
        analyses: list of analysis records
    """
    try:
        session = Session()
        analyses = session.query(MaterialAnalysis).order_by(
            MaterialAnalysis.timestamp.desc()
        ).limit(limit).all()
        
        results = []
        for analysis in analyses:
            results.append({
                'id': analysis.id,
                'timestamp': analysis.timestamp.isoformat(),
                'material_type': analysis.material_type,
                'has_defects': analysis.has_defects,
                'user_label': analysis.user_label
            })
            
        session.close()
        return results
    
    except Exception as e:
        print(f"Error retrieving recent analyses: {e}")
        if 'session' in locals():
            session.close()
        return []

def save_custom_material(name, material_results, defect_results, image=None, notes=None):
    """
    Save a custom material sample to the database
    
    Args:
        name: name for the custom material
        material_results: dict with material classification results
        defect_results: dict with defect detection results
        image: numpy array or PIL Image (optional)
        notes: additional notes (optional)
        
    Returns:
        success: boolean indicating success/failure
    """
    try:
        session = Session()
        
        # Check if name already exists
        existing = session.query(CustomMaterial).filter_by(name=name).first()
        if existing:
            session.close()
            return False, "A custom material with this name already exists"
        
        # Create new custom material
        custom_material = CustomMaterial(
            name=name,
            material_type=material_results['material_type'],
            confidence=material_results['confidence'],
            properties=material_results['properties'],
            has_defects=defect_results['has_defects'],
            defect_types=defect_results['defect_types'],
            defect_locations=defect_results['defect_locations'],
            defect_confidence=defect_results.get('defect_confidence', 0.0),
            notes=notes
        )
        
        # Add image data if provided
        if image is not None:
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_img = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                pil_img = image
            else:
                pil_img = None
                
            # Convert PIL Image to binary data
            if pil_img:
                img_byte_arr = BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                custom_material.image_data = img_byte_arr.getvalue()
        
        # Save to database
        session.add(custom_material)
        session.commit()
        session.close()
        
        return True, "Custom material saved successfully"
    
    except Exception as e:
        print(f"Error saving custom material: {e}")
        if 'session' in locals():
            session.close()
        return False, f"Error: {str(e)}"

def get_custom_materials():
    """
    Get all custom material samples
    
    Returns:
        materials: dict of custom materials (name -> material data)
    """
    try:
        session = Session()
        custom_materials = session.query(CustomMaterial).all()
        
        materials = {}
        for material in custom_materials:
            materials[material.name] = {
                'material_type': material.material_type,
                'confidence': material.confidence,
                'properties': material.properties,
                'has_defects': material.has_defects,
                'defect_types': material.defect_types,
                'defect_locations': material.defect_locations,
                'defect_confidence': material.defect_confidence
            }
            
        session.close()
        return materials
    
    except Exception as e:
        print(f"Error retrieving custom materials: {e}")
        if 'session' in locals():
            session.close()
        return {}

def get_custom_material_image(name):
    """
    Get the image for a custom material sample
    
    Args:
        name: name of the custom material
        
    Returns:
        image: PIL Image or None if not found
    """
    try:
        session = Session()
        material = session.query(CustomMaterial).filter_by(name=name).first()
        
        if not material or not material.image_data:
            session.close()
            return None
            
        # Convert binary data to PIL Image
        image = Image.open(BytesIO(material.image_data))
        
        session.close()
        return image
    
    except Exception as e:
        print(f"Error retrieving custom material image: {e}")
        if 'session' in locals():
            session.close()
        return None

# Initialize the database when the module is imported
init_db()