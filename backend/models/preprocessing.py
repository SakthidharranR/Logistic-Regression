"""
Data preprocessing utilities for logistic regression.
"""

import numpy as np


def flatten_images(images):
    """
    Flatten images from (num_examples, height, width, channels) to (num_features, num_examples)
    
    Arguments:
    images -- numpy array of shape (num_examples, height, width, channels)
    
    Returns:
    flattened -- numpy array of shape (num_features, num_examples)
    """
    num_examples = images.shape[0]
    flattened = images.reshape(num_examples, -1).T
    return flattened


def normalize_images(images):
    """
    Normalize images by dividing by 255.0
    
    Arguments:
    images -- numpy array of any shape
    
    Returns:
    normalized -- numpy array of same shape, normalized to [0, 1]
    """
    normalized = images / 255.0
    return normalized


def get_dataset_info(images):
    """
    Get dataset information (number of training examples, test examples, image dimensions)
    
    Arguments:
    images -- numpy array of shape (num_examples, height, width, channels)
    
    Returns:
    info -- dictionary with keys: m_train, m_test, num_px
    """
    m_train = images.shape[0]
    num_px = images.shape[1]  # Assuming square images (height == width)
    
    # Note: m_test will be determined when we split the data
    # This function just returns the shape info
    info = {
        "m_train": m_train,
        "num_px": num_px
    }
    
    return info


def prepare_training_data(images, labels, num_train, num_test=1000):
    """
    Prepare training and test data by splitting and preprocessing.
    
    Arguments:
    images -- numpy array of shape (num_examples, height, width, channels)
    labels -- numpy array of shape (num_examples,) with labels (0=cat, 1=dog)
    num_train -- number of training examples to use (taken from beginning)
    num_test -- number of test examples to use (taken from end, default 1000)
    
    Returns:
    X_train -- training images, shape (num_features, m_train)
    Y_train -- training labels, shape (1, m_train)
    X_test -- test images, shape (num_features, m_test)
    Y_test -- test labels, shape (1, m_test)
    classes -- tuple of class names ("non-cat", "cat")
    """
    # Split data: first num_train for training, last num_test for testing
    train_images = images[:num_train]
    test_images = images[-num_test:]
    
    train_labels = labels[:num_train] if labels is not None else None
    test_labels = labels[-num_test:] if labels is not None else None
    
    # Flatten images
    train_set_x_flatten = flatten_images(train_images)
    test_set_x_flatten = flatten_images(test_images)
    
    # Normalize
    train_set_x = normalize_images(train_set_x_flatten)
    test_set_x = normalize_images(test_set_x_flatten)
    
    # Reshape labels to (1, m) format
    if train_labels is not None:
        Y_train = train_labels.reshape(1, -1)
        print(f"[PREPROCESSING] Using actual labels for training: {train_labels[:min(10, len(train_labels))]}...")
    else:
        # If no labels, assume all are cats (0)
        Y_train = np.zeros((1, num_train))
        print(f"[PREPROCESSING] WARNING: No labels found! Setting all training labels to 0 (cats)")
    
    if test_labels is not None:
        Y_test = test_labels.reshape(1, -1)
        print(f"[PREPROCESSING] Using actual labels for testing: {test_labels[:min(10, len(test_labels))]}...")
    else:
        # If no labels, assume all are cats (0)
        Y_test = np.zeros((1, num_test))
        print(f"[PREPROCESSING] WARNING: No labels found! Setting all test labels to 0 (cats)")
    
    classes = ("non-cat", "cat")  # 0 = non-cat, 1 = cat (but in our case 0=cat, 1=dog)
    # Actually, for our dataset: 0 = cat, 1 = dog
    # But the model expects: 0 = non-cat, 1 = cat
    # So we need to keep labels as is (0=cat, 1=dog) but interpret differently
    # For now, we'll keep the original labels
    
    return train_set_x, Y_train, test_set_x, Y_test, classes

