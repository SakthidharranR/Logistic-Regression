"""
Example script showing how to use the dataset utilities.
"""

from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.dataset_utils import (
    process_image_directory,
    load_from_hdf5,
    load_from_npz,
    preprocess_image
)

# Example 1: Process a directory of images and save to HDF5
def example_process_directory():
    """Process all images in a directory."""
    input_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    output_path = Path(__file__).parent.parent.parent / "data" / "processed" / "cats.h5"
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("Please add your cat images to the data/raw/ directory")
        return
    
    # Process and save to HDF5
    num_images = process_image_directory(
        input_dir=input_dir,
        output_path=output_path,
        format='hdf5',
        target_size=(64, 64)
    )
    print(f"Processed {num_images} images")


# Example 2: Load dataset from HDF5
def example_load_hdf5():
    """Load images from HDF5 file."""
    hdf5_path = Path(__file__).parent.parent.parent / "data" / "processed" / "cats.h5"
    
    if not hdf5_path.exists():
        print(f"HDF5 file not found: {hdf5_path}")
        return
    
    images, labels, metadata = load_from_hdf5(hdf5_path)
    
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images.shape}")
    print(f"Metadata: {metadata}")
    
    # Example: Get first image
    first_image = images[0]
    print(f"First image shape: {first_image.shape}")
    print(f"First image value range: [{first_image.min():.3f}, {first_image.max():.3f}]")


# Example 3: Load dataset from NPZ
def example_load_npz():
    """Load images from NPZ file."""
    npz_path = Path(__file__).parent.parent.parent / "data" / "processed" / "cats.npz"
    
    if not npz_path.exists():
        print(f"NPZ file not found: {npz_path}")
        return
    
    images, labels, metadata = load_from_npz(npz_path)
    
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images.shape}")


# Example 4: Preprocess a single image
def example_preprocess_single():
    """Preprocess a single image."""
    # Replace with path to your image
    image_path = Path(__file__).parent.parent.parent / "data" / "raw" / "cat1.jpg"
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    
    processed = preprocess_image(image_path, target_size=(64, 64))
    print(f"Processed image shape: {processed.shape}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")


if __name__ == "__main__":
    print("=" * 50)
    print("Dataset Utilities Examples")
    print("=" * 50)
    
    print("\n1. Processing directory of images:")
    example_process_directory()
    
    print("\n2. Loading from HDF5:")
    example_load_hdf5()
    
    print("\n3. Loading from NPZ:")
    example_load_npz()
    
    print("\n4. Preprocessing single image:")
    example_preprocess_single()

