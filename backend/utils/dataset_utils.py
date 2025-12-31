"""
Dataset utilities for storing and loading 64x64x3 cat images.
Supports HDF5 and NumPy formats for efficient storage.
"""

import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import cv2
from typing import Tuple, Optional, Union


def preprocess_image(image_path: Union[str, Path], target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Preprocess a single image to 64x64x3 format.
    
    Args:
        image_path: Path to the image file
        target_size: Target size (width, height). Default: (64, 64)
    
    Returns:
        Preprocessed image as numpy array (64, 64, 3) with values in [0, 1]
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Ensure shape is (64, 64, 3)
    if img_normalized.shape != (*target_size, 3):
        raise ValueError(f"Image shape mismatch: {img_normalized.shape} != ({target_size[0]}, {target_size[1]}, 3)")
    
    return img_normalized


def save_to_hdf5(images: np.ndarray, labels: Optional[np.ndarray], output_path: Union[str, Path], 
                 metadata: Optional[dict] = None):
    """
    Save images to HDF5 format (recommended for large datasets).
    
    Args:
        images: Array of images, shape (N, 64, 64, 3)
        labels: Optional array of labels, shape (N,)
        output_path: Path to save HDF5 file
        metadata: Optional dictionary of metadata to store
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Store images
        f.create_dataset('images', data=images, compression='gzip', compression_opts=9)
        
        # Store labels if provided
        if labels is not None:
            f.create_dataset('labels', data=labels)
        
        # Store metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    f.attrs[key] = value
                else:
                    f.attrs[key] = str(value)
        
        # Store image shape info
        f.attrs['image_shape'] = images.shape[1:]  # (64, 64, 3)
        f.attrs['num_images'] = len(images)


def load_from_hdf5(file_path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Load images from HDF5 format.
    
    Args:
        file_path: Path to HDF5 file
    
    Returns:
        Tuple of (images, labels, metadata)
    """
    with h5py.File(file_path, 'r') as f:
        images = np.array(f['images'])
        labels = np.array(f['labels']) if 'labels' in f else None
        
        # Load metadata
        metadata = dict(f.attrs)
    
    return images, labels, metadata


def save_to_npz(images: np.ndarray, labels: Optional[np.ndarray], output_path: Union[str, Path],
                metadata: Optional[dict] = None):
    """
    Save images to NumPy NPZ format (good for smaller datasets).
    
    Args:
        images: Array of images, shape (N, 64, 64, 3)
        labels: Optional array of labels, shape (N,)
        output_path: Path to save NPZ file
        metadata: Optional dictionary of metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {'images': images}
    if labels is not None:
        save_dict['labels'] = labels
    if metadata:
        save_dict['metadata'] = metadata
    
    np.savez_compressed(output_path, **save_dict)


def load_from_npz(file_path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[dict]]:
    """
    Load images from NumPy NPZ format.
    
    Args:
        file_path: Path to NPZ file
    
    Returns:
        Tuple of (images, labels, metadata)
    """
    data = np.load(file_path, allow_pickle=True)
    images = data['images']
    labels = data['labels'] if 'labels' in data else None
    metadata = dict(data['metadata']) if 'metadata' in data else None
    
    return images, labels, metadata


def process_image_directory(input_dir: Union[str, Path], output_path: Union[str, Path],
                           format: str = 'hdf5', target_size: Tuple[int, int] = (64, 64)) -> int:
    """
    Process all images in a directory and save to specified format.
    If input_dir contains 'cats' and 'dogs' subdirectories, processes both with labels.
    Otherwise, processes all images in the directory without labels.
    
    Args:
        input_dir: Directory containing raw images (or subdirectories 'cats' and 'dogs')
        output_path: Path to save processed dataset
        format: 'hdf5' or 'npz'
        target_size: Target image size (width, height)
    
    Returns:
        Number of images processed
    """
    input_dir = Path(input_dir)
    
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Check if we have cats and dogs subdirectories
    cats_dir = input_dir / "cats"
    dogs_dir = input_dir / "dogs"
    has_labels = cats_dir.exists() and dogs_dir.exists()
    
    processed_images = []
    labels = []
    
    if has_labels:
        # Process cats (label = 0) and dogs (label = 1)
        print("Found 'cats' and 'dogs' subdirectories. Processing with labels...")
        
        # Process cat images (label = 0)
        cat_files = [f for f in cats_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in extensions]
        print(f"Processing {len(cat_files)} cat images (label=0)...")
        
        for i, img_path in enumerate(cat_files):
            try:
                img = preprocess_image(img_path, target_size)
                processed_images.append(img)
                labels.append(0)  # Cat = 0
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(cat_files)} cat images...")
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
        
        # Process dog images (label = 1)
        dog_files = [f for f in dogs_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in extensions]
        print(f"Processing {len(dog_files)} dog images (label=1)...")
        
        for i, img_path in enumerate(dog_files):
            try:
                img = preprocess_image(img_path, target_size)
                processed_images.append(img)
                labels.append(1)  # Dog = 1
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(dog_files)} dog images...")
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                continue
        
        print(f"\nTotal: {len(processed_images)} images ({sum(1 for l in labels if l == 0)} cats, {sum(1 for l in labels if l == 1)} dogs)")
    else:
        # Process all images in directory without labels
        image_files = [f for f in input_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in extensions]
        
        if not image_files:
            raise ValueError(f"No image files found in {input_dir}")
        
        print(f"Found {len(image_files)} images. Processing...")
        
        for i, img_path in enumerate(image_files):
            try:
                img = preprocess_image(img_path, target_size)
                processed_images.append(img)
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_files)} images...")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    if not processed_images:
        raise ValueError("No images were successfully processed")
    
    # Convert to numpy arrays
    images_array = np.array(processed_images)
    labels_array = np.array(labels, dtype=np.int32) if labels else None
    
    # Create metadata
    metadata = {
        'num_images': len(processed_images),
        'image_shape': images_array.shape[1:],
        'target_size': target_size,
        'format': format,
        'has_labels': has_labels
    }
    if has_labels:
        metadata['num_cats'] = int(np.sum(labels_array == 0))
        metadata['num_dogs'] = int(np.sum(labels_array == 1))
    
    # Save in requested format
    if format.lower() == 'hdf5':
        save_to_hdf5(images_array, labels_array, output_path, metadata)
    elif format.lower() == 'npz':
        save_to_npz(images_array, labels_array, output_path, metadata)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'hdf5' or 'npz'")
    
    print(f"Successfully saved {len(processed_images)} images to {output_path}")
    return len(processed_images)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python dataset_utils.py <input_dir> <output_path> [format]")
        print("Example: python dataset_utils.py ../data/raw ../data/processed/cats_dogs.h5 hdf5")
        print("Note: If input_dir contains 'cats' and 'dogs' subdirectories, labels will be created automatically")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_path = sys.argv[2]
    format = sys.argv[3] if len(sys.argv) > 3 else 'hdf5'
    
    process_image_directory(input_dir, output_path, format)

