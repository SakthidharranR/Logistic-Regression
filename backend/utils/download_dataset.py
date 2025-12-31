"""
Script to download cat images for the dataset.
Provides multiple options for obtaining cat images.
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_cats_vs_dogs_dataset(output_dir: Path, num_images: int = 100):
    """
    Download cat images from the Cats vs Dogs dataset.
    This is a popular dataset with thousands of cat images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading Cats vs Dogs Dataset")
    print("=" * 60)
    print("\nNote: This downloads from a public dataset source.")
    print("The dataset contains ~12,500 cat images.")
    print("\nFor a quick start, you can also:")
    print("1. Download manually from: https://www.kaggle.com/c/dogs-vs-cats/data")
    print("2. Use the Unsplash API (requires API key)")
    print("3. Use the script to download from a custom URL list")
    print("\n" + "=" * 60)
    
    # Option 1: Direct download link (if available)
    # Note: You may need to download from Kaggle manually
    print("\nTo download from Kaggle:")
    print("1. Go to: https://www.kaggle.com/c/dogs-vs-cats/data")
    print("2. Download 'train.zip'")
    print("3. Extract and copy cat images (files starting with 'cat.') to data/raw/")
    
    return False


def download_from_urls(url_list: list, output_dir: Path, delay: float = 0.5):
    """
    Download images from a list of URLs.
    
    Args:
        url_list: List of image URLs
        output_dir: Directory to save images
        delay: Delay between downloads (seconds) to be respectful
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {len(url_list)} images...")
    
    for i, url in enumerate(tqdm(url_list, desc="Downloading")):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Determine file extension
            ext = '.jpg'
            if 'png' in response.headers.get('content-type', ''):
                ext = '.png'
            
            output_path = output_dir / f"cat_{i+1:04d}{ext}"
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            time.sleep(delay)  # Be respectful to servers
            
        except Exception as e:
            print(f"\nError downloading {url}: {e}")
            continue
    
    print(f"\nDownloaded images saved to: {output_dir}")


def create_sample_url_list() -> list:
    """
    Create a sample list of cat image URLs.
    Note: These are example URLs - you should replace with actual image URLs.
    """
    # Example URLs (replace with actual cat image URLs)
    # You can get URLs from:
    # - Unsplash API (requires API key)
    # - Pexels API (free, requires API key)
    # - Or manually collect URLs
    
    sample_urls = [
        # Add your cat image URLs here
        # Example format:
        # "https://example.com/cat1.jpg",
        # "https://example.com/cat2.jpg",
    ]
    
    return sample_urls


def download_from_unsplash(output_dir: Path, num_images: int = 100, api_key: Optional[str] = None):
    """
    Download cat images from Unsplash API.
    Requires an API key from https://unsplash.com/developers
    """
    if not api_key:
        print("\n" + "=" * 60)
        print("Unsplash API Download")
        print("=" * 60)
        print("\nTo use Unsplash API:")
        print("1. Sign up at: https://unsplash.com/developers")
        print("2. Create an application to get an API key")
        print("3. Run: python download_dataset.py --unsplash --api-key YOUR_KEY")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {num_images} cat images from Unsplash...")
    
    # Unsplash API endpoint
    base_url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {api_key}"}
    
    downloaded = 0
    page = 1
    
    while downloaded < num_images:
        params = {
            "query": "cat",
            "per_page": min(30, num_images - downloaded),
            "page": page
        }
        
        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'results' not in data or len(data['results']) == 0:
                break
            
            for result in data['results']:
                if downloaded >= num_images:
                    break
                
                img_url = result['urls']['regular']
                img_response = requests.get(img_url)
                img_response.raise_for_status()
                
                output_path = output_dir / f"cat_{downloaded+1:04d}.jpg"
                with open(output_path, 'wb') as f:
                    f.write(img_response.content)
                
                downloaded += 1
                time.sleep(0.5)  # Rate limiting
            
            page += 1
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"\nDownloaded {downloaded} images to {output_dir}")
    return True


def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download cat images for dataset")
    parser.add_argument("--output", "-o", type=str, default="data/raw",
                       help="Output directory (default: data/raw)")
    parser.add_argument("--num-images", "-n", type=int, default=100,
                       help="Number of images to download (default: 100)")
    parser.add_argument("--unsplash", action="store_true",
                       help="Download from Unsplash API")
    parser.add_argument("--api-key", type=str,
                       help="API key for Unsplash")
    parser.add_argument("--urls", type=str,
                       help="Path to text file with image URLs (one per line)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    print("=" * 60)
    print("Cat Image Dataset Downloader")
    print("=" * 60)
    
    if args.unsplash:
        download_from_unsplash(output_dir, args.num_images, args.api_key)
    elif args.urls:
        with open(args.urls, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        download_from_urls(urls, output_dir)
    else:
        print("\n" + "=" * 60)
        print("Download Options:")
        print("=" * 60)
        print("\n1. MANUAL DOWNLOAD (Recommended for beginners):")
        print("   - Go to: https://www.kaggle.com/datasets/salader/dogs-vs-cats")
        print("   - Download the dataset")
        print("   - Extract and copy cat images to data/raw/")
        print("\n2. UNSPLASH API (Free, requires signup):")
        print("   - Sign up: https://unsplash.com/developers")
        print("   - Get API key")
        print("   - Run: python download_dataset.py --unsplash --api-key YOUR_KEY")
        print("\n3. CUSTOM URL LIST:")
        print("   - Create a text file with image URLs (one per line)")
        print("   - Run: python download_dataset.py --urls urls.txt")
        print("\n4. PEXELS API (Free, requires signup):")
        print("   - Sign up: https://www.pexels.com/api/")
        print("   - Similar to Unsplash")
        print("\n" + "=" * 60)
        
        # Show manual instructions
        download_cats_vs_dogs_dataset(output_dir, args.num_images)


if __name__ == "__main__":
    main()

