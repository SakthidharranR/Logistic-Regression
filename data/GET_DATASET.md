# How to Get Cat Images for Your Dataset

## Option 1: Kaggle Cats vs Dogs Dataset (Recommended - Easiest)

**Best for beginners - No API keys needed!**

1. **Sign up for Kaggle** (free): https://www.kaggle.com/
2. **Go to the dataset**: https://www.kaggle.com/datasets/salader/dogs-vs-cats
3. **Download the dataset**:
   - Click "Download" button
   - Extract the ZIP file
4. **Copy cat images**:
   ```bash
   # After extracting, find the train folder
   # Copy all files starting with "cat." to data/raw/
   cp /path/to/extracted/train/cat.* data/raw/
   ```
5. **Process the images**:
   ```bash
   cd backend
   python utils/dataset_utils.py ../data/raw ../data/processed/cats.h5 hdf5
   ```

**Result**: You'll get ~12,500 cat images ready for training!

---

## Option 2: Unsplash API (Free, High Quality)

**Best for curated, high-quality images**

1. **Sign up**: https://unsplash.com/developers
2. **Create an application** to get your API key
3. **Run the download script**:
   ```bash
   cd backend
   python utils/download_dataset.py --unsplash --api-key YOUR_API_KEY --num-images 500
   ```

**Pros**: High quality, diverse images
**Cons**: Requires API key setup

---

## Option 3: Pexels API (Free, Easy)

**Similar to Unsplash**

1. **Sign up**: https://www.pexels.com/api/
2. **Get your API key**
3. **Use their Python library** or modify the download script

---

## Option 4: Manual Collection

**For specific types of cat images**

1. **Collect image URLs** from:
   - Google Images (use browser extensions)
   - Cat image websites
   - Your own photos
2. **Create a text file** (`urls.txt`):
   ```
   https://example.com/cat1.jpg
   https://example.com/cat2.jpg
   ...
   ```
3. **Download**:
   ```bash
   python utils/download_dataset.py --urls urls.txt
   ```

---

## Option 5: Use Pre-processed Datasets

**If you want ready-to-use datasets:**

1. **CIFAR-10** (has some animal images, but not just cats)
2. **ImageNet** (very large, requires processing)
3. **Custom datasets** from research papers

---

## Quick Start (Recommended Path)

```bash
# 1. Create data directories
mkdir -p data/raw data/processed

# 2. Download from Kaggle (manual) or use API
# For Kaggle: Download and extract, then:
cp /path/to/kaggle/train/cat.* data/raw/

# 3. Process images to 64x64x3 format
cd backend
source venv/bin/activate
pip install -r requirements.txt
python utils/dataset_utils.py ../data/raw ../data/processed/cats.h5 hdf5

# 4. Your dataset is ready!
# Load it in your training code:
python -c "from utils.dataset_utils import load_from_hdf5; images, _, _ = load_from_hdf5('../data/processed/cats.h5'); print(f'Loaded {len(images)} images, shape: {images.shape}')"
```

---

## Tips

- **Start small**: Begin with 100-500 images for testing
- **Quality over quantity**: Better to have 1000 good images than 5000 poor ones
- **Diversity**: Try to get cats in different poses, lighting, backgrounds
- **Consistent format**: The script automatically handles resizing to 64x64x3

---

## Need Help?

If you're having trouble:
1. Check that images are in `data/raw/` directory
2. Ensure images are valid (jpg, png, etc.)
3. Run the example script: `python backend/examples/example_dataset_usage.py`

