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
from models import model, prepare_training_data

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
_labels_cache = None
_dataset_loaded = False


@app.on_event("startup")
async def load_dataset_on_startup():
    """Load the dataset into memory on server startup for fast access."""
    global _images_cache, _labels_cache, _dataset_loaded
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
            _images_cache, _labels_cache, metadata = load_from_hdf5(hdf5_path)
            _dataset_loaded = True
            print(f"✓ Loaded {len(_images_cache)} images into memory")
            if _labels_cache is not None:
                import numpy as np
                unique_labels, counts = np.unique(_labels_cache, return_counts=True)
                print(f"✓ Loaded labels: {len(_labels_cache)} labels")
                print(f"  Label distribution: {dict(zip(unique_labels, counts))}")
                print(f"  (0 = cat, 1 = dog)")
            else:
                print(f"⚠ WARNING: No labels found in dataset! All predictions will be 100% because all labels are set to 0")
        except Exception as e:
            print(f"✗ Error loading HDF5 dataset on startup: {e}")
            import traceback
            traceback.print_exc()
            _images_cache = None
            _labels_cache = None
            _dataset_loaded = False
    elif npz_path.exists():
        try:
            print(f"Loading dataset from {npz_path}...")
            _images_cache, _labels_cache, metadata = load_from_npz(npz_path)
            _dataset_loaded = True
            print(f"✓ Loaded {len(_images_cache)} images into memory")
            if _labels_cache is not None:
                import numpy as np
                unique_labels, counts = np.unique(_labels_cache, return_counts=True)
                print(f"✓ Loaded labels: {len(_labels_cache)} labels")
                print(f"  Label distribution: {dict(zip(unique_labels, counts))}")
                print(f"  (0 = cat, 1 = dog)")
            else:
                print(f"⚠ WARNING: No labels found in dataset! All predictions will be 100% because all labels are set to 0")
        except Exception as e:
            print(f"✗ Error loading NPZ dataset on startup: {e}")
            import traceback
            traceback.print_exc()
            _images_cache = None
            _labels_cache = None
            _dataset_loaded = False
    else:
        print("No processed dataset found. Images will be loaded on-demand from raw directory.")


class TrainingConfig(BaseModel):
    num_images: int
    learning_rate: float = 0.005
    num_iterations: int = 2000
    num_test: int = 1000
    num_training_cats: Optional[int] = None
    num_training_dogs: Optional[int] = None
    num_test_cats: Optional[int] = None
    num_test_dogs: Optional[int] = None


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
    learning_rate = config.learning_rate
    num_iterations = config.num_iterations
    num_test = config.num_test
    num_training_cats = config.num_training_cats
    num_training_dogs = config.num_training_dogs
    num_test_cats = config.num_test_cats
    num_test_dogs = config.num_test_dogs
    
    # Validate number
    if num_images < 1 or num_images > 11000:
        raise HTTPException(
            status_code=400, 
            detail="Number of images must be between 1 and 11,000"
        )
    
    # Validate hyperparameters
    if learning_rate <= 0 or learning_rate > 1:
        raise HTTPException(
            status_code=400,
            detail="Learning rate must be between 0 and 1"
        )
    if num_iterations < 1 or num_iterations > 10000:
        raise HTTPException(
            status_code=400,
            detail="Number of iterations must be between 1 and 10,000"
        )
    if num_test < 1 or num_test > 5000:
        raise HTTPException(
            status_code=400,
            detail="Number of test images must be between 1 and 5,000"
        )
    
    # Use cached dataset if available (loaded on startup)
    global _images_cache, _labels_cache, _dataset_loaded
    
    images_array = None
    labels_array = None
    total_available = 0
    
    if _dataset_loaded and _images_cache is not None:
        # Use cached dataset (fastest option)
        images_array = _images_cache
        labels_array = _labels_cache
        total_available = len(images_array)
    else:
        # Fallback: Load from processed dataset
        hdf5_path = DATA_PROCESSED_DIR / "cats_dogs.h5"
        if not hdf5_path.exists():
            hdf5_path = DATA_PROCESSED_DIR / "cats.h5"
        npz_path = DATA_PROCESSED_DIR / "cats_dogs.npz"
        if not npz_path.exists():
            npz_path = DATA_PROCESSED_DIR / "cats.npz"
        
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
    
    # Check if we have enough images (need at least num_images for training + num_test for testing)
    if total_available < num_images + num_test:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough images. Need at least {num_images + num_test} images ({num_images} for training + {num_test} for testing). Available: {total_available}"
        )
    
    # Check if we have images_array loaded
    if images_array is None:
        raise HTTPException(
            status_code=404,
            detail="No processed dataset found. Please process images first using utils/dataset_utils.py"
        )
    
    # Initialize variables
    num_cats_train = None
    num_dogs_train = None
    num_cats_test = None
    num_dogs_test = None
    
    # Get the requested number of images for training
    # Reserve last 1,000 images for testing
    try:
        # IMPORTANT: Shuffle the dataset first to mix cats and dogs
        # Otherwise we might get all cats in training and all dogs in testing
        if labels_array is not None:
            # Create a random permutation to shuffle both images and labels together
            permutation = np.random.permutation(len(images_array))
            images_array = images_array[permutation]
            labels_array = labels_array[permutation]
            print(f"[SHUFFLE] Shuffled dataset to mix cats and dogs")
            
            # Check if specific cat/dog counts are requested
            use_custom_split = (num_training_cats is not None or num_training_dogs is not None or 
                              num_test_cats is not None or num_test_dogs is not None)
            
            if use_custom_split:
                # Select specific numbers of cats and dogs
                # Get indices for cats (label=0) and dogs (label=1)
                cat_indices = np.where(labels_array == 0)[0]
                dog_indices = np.where(labels_array == 1)[0]
                
                # Determine counts (use provided or calculate balanced)
                if num_training_cats is None:
                    num_training_cats = num_images // 2 if num_training_dogs is None else (num_images - (num_training_dogs or 0))
                if num_training_dogs is None:
                    num_training_dogs = num_images - num_training_cats
                if num_test_cats is None:
                    num_test_cats = num_test // 2 if num_test_dogs is None else (num_test - (num_test_dogs or 0))
                if num_test_dogs is None:
                    num_test_dogs = num_test - num_test_cats
                
                # Validate counts
                if num_training_cats + num_training_dogs != num_images:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Training cats ({num_training_cats}) + dogs ({num_training_dogs}) must equal training images ({num_images})"
                    )
                if num_test_cats + num_test_dogs != num_test:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Test cats ({num_test_cats}) + dogs ({num_test_dogs}) must equal test images ({num_test})"
                    )
                if num_training_cats > len(cat_indices) or num_training_dogs > len(dog_indices):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Not enough images. Available: {len(cat_indices)} cats, {len(dog_indices)} dogs. Requested training: {num_training_cats} cats, {num_training_dogs} dogs"
                    )
                if num_test_cats > len(cat_indices) or num_test_dogs > len(dog_indices):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Not enough images. Available: {len(cat_indices)} cats, {len(dog_indices)} dogs. Requested test: {num_test_cats} cats, {num_test_dogs} dogs"
                    )
                if num_training_cats + num_test_cats > len(cat_indices) or num_training_dogs + num_test_dogs > len(dog_indices):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Not enough images. Need {num_training_cats + num_test_cats} cats (have {len(cat_indices)}) and {num_training_dogs + num_test_dogs} dogs (have {len(dog_indices)})"
                    )
                
                # Select training images
                train_cat_indices = cat_indices[:num_training_cats]
                train_dog_indices = dog_indices[:num_training_dogs]
                training_indices = np.concatenate([train_cat_indices, train_dog_indices])
                np.random.shuffle(training_indices)  # Shuffle to mix cats and dogs
                
                # Select test images (from remaining)
                remaining_cat_indices = cat_indices[num_training_cats:]
                remaining_dog_indices = dog_indices[num_training_dogs:]
                test_cat_indices = remaining_cat_indices[:num_test_cats]
                test_dog_indices = remaining_dog_indices[:num_test_dogs]
                test_indices = np.concatenate([test_cat_indices, test_dog_indices])
                np.random.shuffle(test_indices)  # Shuffle to mix cats and dogs
                
                training_images = images_array[training_indices]
                test_images = images_array[test_indices]
                training_labels = labels_array[training_indices]
                test_labels = labels_array[test_indices]
                
                num_cats_train = num_training_cats
                num_dogs_train = num_training_dogs
                num_cats_test = num_test_cats
                num_dogs_test = num_test_dogs
                
                print(f"[CUSTOM SPLIT] Training: {num_training_cats} cats, {num_training_dogs} dogs")
                print(f"[CUSTOM SPLIT] Test: {num_test_cats} cats, {num_test_dogs} dogs")
            else:
                # Use default balanced approach
                # Use first num_images for training, last num_test for testing
                training_images = images_array[:num_images]
                test_images = images_array[-num_test:]
                
                training_labels = labels_array[:num_images]
                test_labels = labels_array[-num_test:]
                
                # Count cats and dogs in training and test sets
                num_cats_train = int(np.sum(training_labels == 0))
                num_dogs_train = int(np.sum(training_labels == 1))
                num_cats_test = int(np.sum(test_labels == 0))
                num_dogs_test = int(np.sum(test_labels == 1))
        else:
            # No labels available, use simple split
            training_images = images_array[:num_images]
            test_images = images_array[-num_test:]
            training_labels = None
            test_labels = None
        
        # Count cats and dogs in training and test sets
        if training_labels is not None:
            num_cats_train = int(np.sum(training_labels == 0))
            num_dogs_train = int(np.sum(training_labels == 1))
        if test_labels is not None:
            num_cats_test = int(np.sum(test_labels == 0))
            num_dogs_test = int(np.sum(test_labels == 1))
        
        # Prepare training data using preprocessing functions
        # Collect debug info for response
        debug_info = []
        
        print(f"\n{'='*60}")
        print(f"=== TRAINING DEBUG INFO ===")
        print(f"{'='*60}")
        debug_info.append("=== TRAINING DEBUG INFO ===")
        
        print(f"Training with {num_images} images, testing with {num_test} images")
        debug_info.append(f"Training with {num_images} images, testing with {num_test} images")
        
        print(f"Labels available: {labels_array is not None}")
        debug_info.append(f"Labels available: {labels_array is not None}")
        
        if labels_array is not None:
            print(f"Total labels shape: {labels_array.shape}")
            debug_info.append(f"Total labels shape: {labels_array.shape}")
            
            train_labels_sample = labels_array[:min(num_images, 20)]
            test_labels_sample = labels_array[-min(num_test, 20):]
            print(f"Training labels (first {min(num_images, 20)}): {train_labels_sample}")
            debug_info.append(f"Training labels (first {min(num_images, 20)}): {train_labels_sample.tolist()}")
            print(f"Test labels (last {min(num_test, 20)}): {test_labels_sample}")
            debug_info.append(f"Test labels (last {min(num_test, 20)}): {test_labels_sample.tolist()}")
            
            unique_train = np.unique(labels_array[:num_images])
            unique_test = np.unique(labels_array[-num_test:])
            print(f"Unique labels in training set: {unique_train}")
            debug_info.append(f"Unique labels in training set: {unique_train.tolist()}")
            print(f"Unique labels in test set: {unique_test}")
            debug_info.append(f"Unique labels in test set: {unique_test.tolist()}")
            
            train_cats = int(np.sum(labels_array[:num_images] == 0))
            train_dogs = int(np.sum(labels_array[:num_images] == 1))
            test_cats = int(np.sum(labels_array[-num_test:] == 0))
            test_dogs = int(np.sum(labels_array[-num_test:] == 1))
            print(f"Training set: {train_cats} cats, {train_dogs} dogs")
            debug_info.append(f"Training set: {train_cats} cats, {train_dogs} dogs")
            print(f"Test set: {test_cats} cats, {test_dogs} dogs")
            debug_info.append(f"Test set: {test_cats} cats, {test_dogs} dogs")
        else:
            warning_msg = "⚠⚠⚠ WARNING: No labels found in dataset! ⚠⚠⚠"
            print(warning_msg)
            debug_info.append(warning_msg)
            print("All labels will be set to 0 (cats), causing 100% accuracy!")
            debug_info.append("All labels will be set to 0 (cats), causing 100% accuracy!")
            print("You need to reprocess the dataset with cats/ and dogs/ subdirectories.")
            debug_info.append("You need to reprocess the dataset with cats/ and dogs/ subdirectories.")
            print("Run: cd backend && python utils/dataset_utils.py ../data/raw ../data/processed/cats_dogs.h5 hdf5")
            debug_info.append("Run: cd backend && python utils/dataset_utils.py ../data/raw ../data/processed/cats_dogs.h5 hdf5")
        
        X_train, Y_train, X_test, Y_test, classes = prepare_training_data(
            images_array, 
            labels_array, 
            num_train=num_images, 
            num_test=num_test
        )
        
        print(f"\nAfter preprocessing:")
        print(f"X_train shape: {X_train.shape}")
        print(f"Y_train shape: {Y_train.shape}, values: {Y_train}")
        print(f"X_test shape: {X_test.shape}")
        print(f"Y_test shape: {Y_test.shape}, unique values: {np.unique(Y_test)}")
        print(f"Y_train unique values: {np.unique(Y_train)}")
        
        # Train the logistic regression model
        # Using default hyperparameters: num_iterations=2000, learning_rate=0.005
        print(f"\nStarting training with {2000} iterations, learning_rate=0.005...")
        logistic_regression_model = model(
            X_train, 
            Y_train, 
            X_test, 
            Y_test, 
            num_iterations=2000, 
            learning_rate=0.005, 
            print_cost=True  # Print costs to see training progress
        )
        
        print(f"\n=== PREDICTION DEBUG ===")
        Y_pred_train = logistic_regression_model["Y_prediction_train"]
        Y_pred_test = logistic_regression_model["Y_prediction_test"]
        
        print(f"Y_train (actual): {Y_train}")
        print(f"Y_pred_train (predicted): {Y_pred_train}")
        print(f"Y_test (actual): {Y_test}")
        print(f"Y_pred_test (predicted): {Y_pred_test}")
        
        print(f"\nTraining set:")
        print(f"  Actual labels: {Y_train.flatten()}")
        print(f"  Predicted labels: {Y_pred_train.flatten()}")
        print(f"  Matches: {Y_train.flatten() == Y_pred_train.flatten()}")
        
        print(f"\nTest set (first 20):")
        print(f"  Actual labels: {Y_test.flatten()[:20]}")
        print(f"  Predicted labels: {Y_pred_test.flatten()[:20]}")
        print(f"  Matches: {Y_test.flatten()[:20] == Y_pred_test.flatten()[:20]}")
        
        # Calculate accuracies
        train_accuracy = 100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100
        test_accuracy = 100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100
        
        print(f"\n=== ACCURACY CALCULATION ===")
        print(f"Train accuracy: {train_accuracy}%")
        print(f"Test accuracy: {test_accuracy}%")
        train_errors = int(np.sum(Y_pred_train != Y_train))
        test_errors = int(np.sum(Y_pred_test != Y_test))
        print(f"Train errors: {train_errors} out of {Y_train.shape[1]}")
        print(f"Test errors: {test_errors} out of {Y_test.shape[1]}")
        print(f"===========================\n")
        
        # Add accuracy info to debug logs
        debug_info.append(f"\n=== ACCURACY CALCULATION ===")
        debug_info.append(f"Train accuracy: {train_accuracy:.2f}%")
        debug_info.append(f"Test accuracy: {test_accuracy:.2f}%")
        debug_info.append(f"Train errors: {train_errors} out of {Y_train.shape[1]}")
        debug_info.append(f"Test errors: {test_errors} out of {Y_test.shape[1]}")
        
        # Extract costs (convert numpy arrays to lists for JSON serialization)
        costs = [float(cost) for cost in logistic_regression_model["costs"]]
        
        return {
            "status": "training_completed",
            "num_images": num_images,
            "total_available": total_available,
            "training_set_size": num_images,
            "test_set_size": num_test,
            "has_labels": labels_array is not None,
            "training_cats": num_cats_train,
            "training_dogs": num_dogs_train,
            "test_cats": num_cats_test,
            "test_dogs": num_dogs_test,
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "costs": costs,
            "num_iterations": logistic_regression_model["num_iterations"],
            "learning_rate": logistic_regression_model["learning_rate"],
            "debug_logs": debug_info,  # Include debug logs in response
            "message": f"Training completed! Train accuracy: {train_accuracy:.2f}%, Test accuracy: {test_accuracy:.2f}%"
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error during training: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during training: {str(e)}"
        )

