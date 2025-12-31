# Dataset Storage

This directory contains the cat image dataset for training.

## Structure

```
data/
├── raw/              # Original images (before preprocessing)
├── processed/        # Preprocessed 64x64x3 images
│   ├── cats.h5       # HDF5 format (recommended for large datasets)
│   └── cats.npz      # NumPy compressed format (alternative)
└── README.md
```

## Storage Formats

### HDF5 (Recommended)
- **File**: `cats.h5`
- **Advantages**: 
  - Efficient compression
  - Fast I/O for large datasets
  - Can store metadata
  - Supports partial loading
- **Use for**: Large datasets (>1000 images)

### NumPy NPZ (Alternative)
- **File**: `cats.npz`
- **Advantages**:
  - Simple to use
  - Native NumPy format
  - Good for smaller datasets
- **Use for**: Smaller datasets (<1000 images)

## Image Specifications

- **Size**: 64x64 pixels
- **Channels**: 3 (RGB)
- **Format**: NumPy array, dtype: float32, normalized [0, 1]

