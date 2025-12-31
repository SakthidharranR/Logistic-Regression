from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from pathlib import Path
import sys
from typing import List, Optional
import base64
import cv2
import h5py

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from utils.dataset_utils import load_from_hdf5, load_from_npz

app = FastAPI(title="AI Calculation API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port and React default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Global variable to cache the loaded dataset in memory
_images_cache = None
_dataset_loaded = False


@app.on_event("startup")
async def load_dataset_on_startup():
    """Load the dataset into memory on server startup for fast access."""
    global _images_cache, _dataset_loaded
    # Try cats_dogs.h5 first (new format with labels), then fallback to cats.h5
    hdf5_path = DATA_PROCESSED_DIR / "cats_dogs.h5"
    if not hdf5_path.exists():
        hdf5_path = DATA_PROCESSED_DIR / "cats.h5"
    npz_path = DATA_PROCESSED_DIR / "cats_dogs.npz"
    if not npz_path.exists():
        npz_path = DATA_PROCESSED_DIR / "cats.npz"
    
    if hdf5_path.exists():
        try:
            print(f"Loading dataset from {hdf5_path}...")
            _images_cache, _, _ = load_from_hdf5(hdf5_path)
            _dataset_loaded = True
            print(f"✓ Loaded {len(_images_cache)} images into memory")
        except Exception as e:
            print(f"✗ Error loading HDF5 dataset on startup: {e}")
            _images_cache = None
            _dataset_loaded = False
    elif npz_path.exists():
        try:
            print(f"Loading dataset from {npz_path}...")
            _images_cache, _, _ = load_from_npz(npz_path)
            _dataset_loaded = True
            print(f"✓ Loaded {len(_images_cache)} images into memory")
        except Exception as e:
            print(f"✗ Error loading NPZ dataset on startup: {e}")
            _images_cache = None
            _dataset_loaded = False
    else:
        print("No processed dataset found. Images will be loaded on-demand from raw directory.")


class TrainingConfig(BaseModel):
    num_images: int


@app.get("/")
def read_root():
    return {"message": "AI Calculation API is running"}


@app.get("/api/images/list")
def get_image_list(num_images: Optional[int] = None):
    """
    Get list of available images from the processed dataset.
    If num_images is specified, returns that many images.
    By default, returns just the total count without listing all indices.
    """
    # Try cats_dogs.h5 first (new format with labels), then fallback to cats.h5
    hdf5_path = DATA_PROCESSED_DIR / "cats_dogs.h5"
    if not hdf5_path.exists():
        hdf5_path = DATA_PROCESSED_DIR / "cats.h5"
    npz_path = DATA_PROCESSED_DIR / "cats_dogs.npz"
    if not npz_path.exists():
        npz_path = DATA_PROCESSED_DIR / "cats.npz"
    
    # Try to load from processed dataset first
    images_array = None
    total_count = 0
    
    if hdf5_path.exists():
        try:
            # Only load metadata to get count, don't load all images
            with h5py.File(hdf5_path, 'r') as f:
                if 'images' in f:
                    total_count = f['images'].shape[0]
                else:
                    # Try to get from metadata
                    total_count = f.attrs.get('num_images', 0)
            images_array = None  # Don't load all images into memory
        except Exception as e:
            return {"error": f"Error loading HDF5 dataset: {str(e)}", "count": 0}
    elif npz_path.exists():
        try:
            # For NPZ, we need to load to get count, but it's smaller
            data = np.load(npz_path, allow_pickle=True)
            total_count = len(data['images']) if 'images' in data else 0
            images_array = None  # Don't keep in memory
        except Exception as e:
            return {"error": f"Error loading NPZ dataset: {str(e)}", "count": 0}
    else:
        # Fallback to raw images if no processed dataset
        if not DATA_RAW_DIR.exists():
            return {
                "images": [], 
                "count": 0, 
                "message": "No processed dataset found. Please process images first using utils/dataset_utils.py"
            }
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f.name for f in DATA_RAW_DIR.iterdir() 
            if f.is_file() and f.suffix.lower() in extensions
        ]
        image_files.sort()
        
        if num_images:
            image_files = image_files[:num_images]
        
        return {
            "images": image_files,
            "count": len(image_files),
            "total_count": len(image_files),
            "message": f"Found {len(image_files)} images (from raw directory)"
        }
    
    # If we have processed dataset, return response
    if total_count > 0:
        # Only return actual indices if num_images is specified and reasonable (up to 11,000)
        # Otherwise, just return the count to avoid huge JSON responses
        # Frontend will generate indices as needed
        if num_images and num_images <= 11000:
            image_indices = list(range(min(num_images, total_count)))
        else:
            # Don't return all indices - just return count
            # Frontend can generate indices as needed (up to 11,000)
            image_indices = []
        
        return {
            "images": image_indices,  # Empty list if not requesting specific number
            "count": len(image_indices) if image_indices else 0,
            "total_count": total_count,
            "from_dataset": True,
            "message": f"Found {total_count} images from processed dataset"
        }
    
    return {"images": [], "count": 0, "message": "No images found"}


@app.get("/api/images/{image_name}")
def get_image(image_name: str):
    """
    Serve an image file from the raw data directory.
    """
    image_path = DATA_RAW_DIR / image_name
    
    # Security: prevent directory traversal
    if not image_path.exists() or not str(image_path).startswith(str(DATA_RAW_DIR)):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)


@app.get("/api/images/preview/{image_id}")
def get_image_preview(image_id: str, size: int = 64):
    """
    Get a base64-encoded preview of an image from the processed dataset.
    image_id can be an index (for processed dataset) or filename (for raw images).
    Uses cached dataset loaded on startup for fast access.
    """
    global _images_cache, _dataset_loaded
    
    # Try to use cached dataset first (fast path)
    img_array = None
    
    if _dataset_loaded and _images_cache is not None:
        try:
            idx = int(image_id)
            if 0 <= idx < len(_images_cache):
                img_array = _images_cache[idx]
        except ValueError:
            raise HTTPException(status_code=404, detail="Invalid image index")
    else:
        # Fallback: load on-demand if cache not available
        hdf5_path = DATA_PROCESSED_DIR / "cats_dogs.h5"
        if not hdf5_path.exists():
            hdf5_path = DATA_PROCESSED_DIR / "cats.h5"
        npz_path = DATA_PROCESSED_DIR / "cats_dogs.npz"
        if not npz_path.exists():
            npz_path = DATA_PROCESSED_DIR / "cats.npz"
        
        if hdf5_path.exists():
            try:
                images_array, _, _ = load_from_hdf5(hdf5_path)
                try:
                    idx = int(image_id)
                    if 0 <= idx < len(images_array):
                        img_array = images_array[idx]
                except ValueError:
                    raise HTTPException(status_code=404, detail="Invalid image index")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading from HDF5: {str(e)}")
        elif npz_path.exists():
            try:
                images_array, _, _ = load_from_npz(npz_path)
                try:
                    idx = int(image_id)
                    if 0 <= idx < len(images_array):
                        img_array = images_array[idx]
                except ValueError:
                    raise HTTPException(status_code=404, detail="Invalid image index")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error loading from NPZ: {str(e)}")
        else:
            # Fallback to raw images
            image_path = DATA_RAW_DIR / image_id
            if not image_path.exists():
                raise HTTPException(status_code=404, detail="Image not found")
            
            img = cv2.imread(str(image_path))
            if img is None:
                raise HTTPException(status_code=500, detail="Could not load image")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            
            # Encode to base64
            _, buffer = cv2.imencode('.jpg', img_resized)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "image": f"data:image/jpeg;base64,{img_base64}",
                "name": image_id,
                "index": None
            }
    
    # Process image from dataset (already 64x64x3, normalized [0,1])
    if img_array is None:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Convert from normalized [0,1] to [0,255] uint8
    img_uint8 = (img_array * 255).astype(np.uint8)
    
    # Resize for preview (from 64x64 to requested size)
    img_resized = cv2.resize(img_uint8, (size, size), interpolation=cv2.INTER_AREA)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', img_resized)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "image": f"data:image/jpeg;base64,{img_base64}",
        "name": f"cat_{image_id}",
        "index": int(image_id)
    }


@app.get("/api/dataset/info")
def get_dataset_info():
    """
    Get information about available processed datasets.
    """
    # Try cats_dogs.h5 first (new format with labels), then fallback to cats.h5
    hdf5_path = DATA_PROCESSED_DIR / "cats_dogs.h5"
    if not hdf5_path.exists():
        hdf5_path = DATA_PROCESSED_DIR / "cats.h5"
    npz_path = DATA_PROCESSED_DIR / "cats_dogs.npz"
    if not npz_path.exists():
        npz_path = DATA_PROCESSED_DIR / "cats.npz"
    
    info = {
        "hdf5_available": hdf5_path.exists(),
        "npz_available": npz_path.exists(),
        "raw_images_count": 0,
        "has_labels": False
    }
    
    # Count raw images
    if DATA_RAW_DIR.exists():
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in DATA_RAW_DIR.iterdir() 
                      if f.is_file() and f.suffix.lower() in extensions]
        info["raw_images_count"] = len(image_files)
        
        # Check if we have cats and dogs subdirectories
        cats_dir = DATA_RAW_DIR / "cats"
        dogs_dir = DATA_RAW_DIR / "dogs"
        if cats_dir.exists() and dogs_dir.exists():
            cat_files = [f for f in cats_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]
            dog_files = [f for f in dogs_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions]
            info["raw_cats_count"] = len(cat_files)
            info["raw_dogs_count"] = len(dog_files)
    
    # Get processed dataset info if available
    if hdf5_path.exists():
        try:
            _, labels, metadata = load_from_hdf5(hdf5_path)
            info["processed_count"] = metadata.get('num_images', 0)
            info["image_shape"] = metadata.get('image_shape', None)
            info["has_labels"] = labels is not None
            if labels is not None:
                info["num_cats"] = int(np.sum(labels == 0))
                info["num_dogs"] = int(np.sum(labels == 1))
        except Exception as e:
            info["hdf5_error"] = str(e)
    elif npz_path.exists():
        try:
            _, labels, metadata = load_from_npz(npz_path)
            info["processed_count"] = metadata.get('num_images', 0) if metadata else 0
            info["image_shape"] = metadata.get('image_shape', None) if metadata else None
            info["has_labels"] = labels is not None
            if labels is not None:
                info["num_cats"] = int(np.sum(labels == 0))
                info["num_dogs"] = int(np.sum(labels == 1))
        except Exception as e:
            info["npz_error"] = str(e)
    
    return info


@app.post("/api/training/start")
def start_training(config: TrainingConfig):
    """
    Start training with specified number of images from the processed dataset.
    """
    num_images = config.num_images
    
    # Validate number
    if num_images < 2000 or num_images > 11000:
        raise HTTPException(
            status_code=400, 
            detail="Number of images must be between 2,000 and 11,000 (1,000 reserved for testing)"
        )
    
    # Load from processed dataset
    hdf5_path = DATA_PROCESSED_DIR / "cats_dogs.h5"
    if not hdf5_path.exists():
        hdf5_path = DATA_PROCESSED_DIR / "cats.h5"
    npz_path = DATA_PROCESSED_DIR / "cats_dogs.npz"
    if not npz_path.exists():
        npz_path = DATA_PROCESSED_DIR / "cats.npz"
    
    images_array = None
    labels_array = None
    total_available = 0
    
    if hdf5_path.exists():
        try:
            images_array, labels_array, metadata = load_from_hdf5(hdf5_path)
            total_available = len(images_array)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading HDF5 dataset: {str(e)}"
            )
    elif npz_path.exists():
        try:
            images_array, labels_array, metadata = load_from_npz(npz_path)
            total_available = len(images_array)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error loading NPZ dataset: {str(e)}"
            )
    else:
        # Fallback to raw images
        if not DATA_RAW_DIR.exists():
            raise HTTPException(
                status_code=404,
                detail="No processed dataset found. Please process images first using utils/dataset_utils.py"
            )
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in DATA_RAW_DIR.iterdir() 
                      if f.is_file() and f.suffix.lower() in extensions]
        total_available = len(image_files)
    
    # Check if we have enough images
    if total_available < num_images + 1000:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough images. Need at least {num_images + 1000} images (11,000 for training + 1,000 for testing). Available: {total_available}"
        )
    
    # Initialize variables
    num_cats_train = None
    num_dogs_train = None
    num_cats_test = None
    num_dogs_test = None
    
    # Get the requested number of images for training
    # Reserve last 1,000 images for testing
    if images_array is not None:
        # Use first num_images for training, last 1000 for testing
        training_images = images_array[:num_images]
        test_images = images_array[-1000:]
        
        training_labels = labels_array[:num_images] if labels_array is not None else None
        test_labels = labels_array[-1000:] if labels_array is not None else None
        
        # Count cats and dogs in training and test sets
        if training_labels is not None:
            num_cats_train = int(np.sum(training_labels == 0))
            num_dogs_train = int(np.sum(training_labels == 1))
        if test_labels is not None:
            num_cats_test = int(np.sum(test_labels == 0))
            num_dogs_test = int(np.sum(test_labels == 1))
        
        # TODO: Implement actual training logic here
        # training_images shape: (num_images, 64, 64, 3) - ready for training!
        # test_images shape: (1000, 64, 64, 3) - ready for testing!
    
    # TODO: Implement actual training logic here
    # For now, return a placeholder response
    
    return {
        "status": "training_started",
        "num_images": num_images,
        "total_available": total_available,
        "training_set_size": num_images,
        "test_set_size": 1000,
        "has_labels": labels_array is not None,
        "training_cats": num_cats_train,
        "training_dogs": num_dogs_train,
        "test_cats": num_cats_test,
        "test_dogs": num_dogs_test,
        "message": f"Training started with {num_images} images for training and 1,000 images reserved for testing (total available: {total_available})"
    }

