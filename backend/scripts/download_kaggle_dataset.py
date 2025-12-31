"""
Script to download the Microsoft Cats vs Dogs dataset from Kaggle using kagglehub.
"""

import os
import sys
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import kagglehub
except ImportError:
    print("kagglehub not installed. Installing...")
    os.system(f"{sys.executable} -m pip install kagglehub")
    import kagglehub

def download_cats_vs_dogs_dataset(output_dir: Path, api_token: str = None):
    """
    Download the Microsoft Cats vs Dogs dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
        api_token: Kaggle API token (optional, can be set via environment variable)
    """
    # Set up Kaggle credentials if token provided
    if api_token:
        # Create .kaggle directory if it doesn't exist
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        
        # Write kaggle.json file
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text(f'{{"username":"kaggle","key":"{api_token}"}}')
        kaggle_json.chmod(0o600)  # Make it readable only by owner
        
        print(f"Kaggle credentials saved to {kaggle_json}")
    
    print("=" * 60)
    print("Downloading Microsoft Cats vs Dogs Dataset from Kaggle")
    print("=" * 60)
    
    try:
        # Download the dataset
        print("\nDownloading dataset...")
        path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")
        
        print(f"\n✓ Dataset downloaded to: {path}")
        
        # Find cat and dog images in the downloaded dataset
        dataset_path = Path(path)
        print(f"\nLooking for cat and dog images in: {dataset_path}")
        
        # Common locations for images in this dataset
        possible_cat_locations = [
            dataset_path / "PetImages" / "Cat",
            dataset_path / "train" / "Cat",
            dataset_path / "Cat",
            dataset_path / "cats",
        ]
        
        possible_dog_locations = [
            dataset_path / "PetImages" / "Dog",
            dataset_path / "train" / "Dog",
            dataset_path / "Dog",
            dataset_path / "dogs",
        ]
        
        cat_images_dir = None
        dog_images_dir = None
        
        # Find cat directory
        for loc in possible_cat_locations:
            if loc.exists() and loc.is_dir():
                cat_images_dir = loc
                break
        
        # Find dog directory
        for loc in possible_dog_locations:
            if loc.exists() and loc.is_dir():
                dog_images_dir = loc
                break
        
        # Also search recursively
        if cat_images_dir is None or dog_images_dir is None:
            for root, dirs, files in os.walk(dataset_path):
                root_lower = root.lower()
                if "cat" in root_lower and cat_images_dir is None and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    cat_images_dir = Path(root)
                if "dog" in root_lower and dog_images_dir is None and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    dog_images_dir = Path(root)
                if cat_images_dir and dog_images_dir:
                    break
        
        if cat_images_dir is None:
            print("\n⚠ Warning: Could not find cat images directory automatically.")
            print(f"Please manually copy cat images from {dataset_path} to {output_dir / 'cats'}")
        
        if dog_images_dir is None:
            print("\n⚠ Warning: Could not find dog images directory automatically.")
            print(f"Please manually copy dog images from {dataset_path} to {output_dir / 'dogs'}")
        
        if cat_images_dir is None or dog_images_dir is None:
            print("\nDataset structure:")
            for root, dirs, files in os.walk(dataset_path):
                level = root.replace(str(dataset_path), '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
            return False
        
        print(f"\n✓ Found cat images in: {cat_images_dir}")
        print(f"✓ Found dog images in: {dog_images_dir}")
        
        # Count and copy images
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Create subdirectories
        cats_dir = output_dir / "cats"
        dogs_dir = output_dir / "dogs"
        cats_dir.mkdir(parents=True, exist_ok=True)
        dogs_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy cat images
        cat_files = [f for f in cat_images_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in extensions]
        print(f"\nCopying {len(cat_files)} cat images...")
        copied_cats = 0
        for img_file in cat_files:
            try:
                shutil.copy2(img_file, cats_dir / img_file.name)
                copied_cats += 1
                if copied_cats % 500 == 0:
                    print(f"  Copied {copied_cats}/{len(cat_files)} cat images...")
            except Exception as e:
                print(f"  Error copying {img_file.name}: {e}")
        
        # Copy dog images
        dog_files = [f for f in dog_images_dir.iterdir() 
                    if f.is_file() and f.suffix.lower() in extensions]
        print(f"\nCopying {len(dog_files)} dog images...")
        copied_dogs = 0
        for img_file in dog_files:
            try:
                shutil.copy2(img_file, dogs_dir / img_file.name)
                copied_dogs += 1
                if copied_dogs % 500 == 0:
                    print(f"  Copied {copied_dogs}/{len(dog_files)} dog images...")
            except Exception as e:
                print(f"  Error copying {img_file.name}: {e}")
        
        print(f"\n✓ Successfully copied {copied_cats} cat images to {cats_dir}")
        print(f"✓ Successfully copied {copied_dogs} dog images to {dogs_dir}")
        print(f"\nNext step: Process the images:")
        print(f"  cd backend")
        print(f"  source venv/bin/activate")
        print(f"  python utils/dataset_utils.py ../data/raw ../data/processed/cats_dogs.h5 hdf5")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your Kaggle API token is correct")
        print("2. Check your internet connection")
        print("3. Verify the dataset name is correct")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Cats vs Dogs dataset from Kaggle")
    parser.add_argument("--output", "-o", type=str, default="data/raw",
                       help="Output directory for cat and dog images (default: data/raw)")
    parser.add_argument("--token", "-t", type=str,
                       help="Kaggle API token (or set KAGGLE_API_TOKEN environment variable)")
    
    args = parser.parse_args()
    
    # Get token from args or environment
    api_token = args.token or os.environ.get("KAGGLE_API_TOKEN")
    
    if not api_token:
        print("Error: Kaggle API token required!")
        print("Usage: python download_kaggle_dataset.py --token YOUR_TOKEN")
        print("Or set KAGGLE_API_TOKEN environment variable")
        sys.exit(1)
    
    output_dir = Path(__file__).parent.parent.parent / args.output
    
    success = download_cats_vs_dogs_dataset(output_dir, api_token)
    
    if success:
        print("\n" + "=" * 60)
        print("✓ Dataset download complete!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Dataset download failed")
        print("=" * 60)
        sys.exit(1)

