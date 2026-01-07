#!/bin/bash

# Quick setup script for downloading and processing cat images

echo "=========================================="
echo "Cat Dataset Setup Script"
echo "=========================================="
echo ""

# Create directories
echo "Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed

echo "✓ Directories created"
echo ""

echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "OPTION 1 - Kaggle (Recommended):"
echo "  1. Go to: https://www.kaggle.com/datasets/salader/dogs-vs-cats"
echo "  2. Download the dataset"
echo "  3. Extract and copy cat images:"
echo "     cp /path/to/train/cat.* data/raw/"
echo "  4. Process images:"
echo "     cd backend"
echo "     source venv/bin/activate"
echo "     python utils/dataset_utils.py ../data/raw ../data/processed/cats.h5 hdf5"
echo ""
echo "OPTION 2 - Unsplash API:"
echo "  1. Get API key from: https://unsplash.com/developers"
echo "  2. Run:"
echo "     cd backend"
echo "     python utils/download_dataset.py --unsplash --api-key YOUR_KEY --num-images 500"
echo ""
echo "OPTION 3 - Manual:"
echo "  Place your cat images in data/raw/ directory"
echo "  Then process with:"
echo "    cd backend"
echo "    python utils/dataset_utils.py ../data/raw ../data/processed/cats.h5 hdf5"
echo ""
echo "=========================================="
echo "For more details, see: data/GET_DATASET.md"
echo "=========================================="

