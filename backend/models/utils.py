"""
Utility functions for loading dataset from backend cache.
"""

import numpy as np


def load_dataset_from_cache(images_cache, labels_cache):
    """
    Load dataset from backend's cached dataset.
    
    Arguments:
    images_cache -- numpy array of cached images, shape (num_examples, height, width, channels)
    labels_cache -- numpy array of cached labels, shape (num_examples,) or None
    
    Returns:
    train_set_x_orig -- training images, shape (num_train, height, width, channels)
    train_set_y -- training labels, shape (num_train,)
    test_set_x_orig -- test images, shape (num_test, height, width, channels)
    test_set_y -- test labels, shape (num_test,)
    classes -- tuple of class names
    """
    # This function is a wrapper that returns the cached data
    # The actual splitting will be done in prepare_training_data
    # For now, just return the full dataset
    
    if labels_cache is not None:
        classes = ("cat", "dog")  # 0 = cat, 1 = dog
    else:
        classes = ("cat",)  # Only cats if no labels
    
    # Return full dataset - splitting will happen in prepare_training_data
    return images_cache, labels_cache, classes

