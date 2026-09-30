import sys
from collections import Counter
import re
import os
import numpy as np
from PIL import Image

def get_volume_shape(in_path):
    """
    Returns array with shape of volume by given tiffs path.
    """
    # Get all files in directory
    all_files = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
    
    if not all_files:
        raise FileNotFoundError("No files found in the directory")
    
    # Find most common extension
    common_ext = get_most_common_extension(all_files)
    if not common_ext:
        raise ValueError("No files with extensions found in the directory (maybe DICOM).")
    
    print(f"Using files with extension: {common_ext}")
    
    # Filter files by most common extension
    files = [f for f in all_files if os.path.splitext(f)[1].lower() == common_ext]
    
    if not files:
        raise ValueError(f"No files with extension {common_ext} found")
    
    # Natural sort files
    files.sort(key=natural_sort_key)
    
    # Get full paths
    file_paths = [os.path.join(in_path, x) for x in files]
    # Verify first file is a valid image
    try:
        with Image.open(file_paths[0]) as img:
            size_y, size_x = np.array(img).shape
    except Exception as e:
        print(f"Error opening image file '{file_paths[0]}': {e}")
        print(f"Note: Files with extension {common_ext} must be valid image files")
        sys.exit(1)
    print(f'size_y, size_x = {size_y, size_x}')
    size_z = len(files)
    shape = size_z, size_y, size_x
    return shape

def load_volume(files, size_z, size_y, size_x):#, sigma = None):
    """
    Returns npy array with volume.
    """
    volume = np.zeros((size_z, size_y, size_x))
    for slice_number, file in enumerate(files):
        if slice_number%100 == 0:
            print(f'Load slice number {slice_number}/{len(files)}', end = "\r")
        volume[slice_number] = np.array(Image.open(file))
    print(f'Load slice number {len(files)}/{len(files)}')
    return volume

def natural_sort_key(s):
    """
    Key function for natural sorting of strings containing numbers.
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]

def get_most_common_extension(files):
    """
    Returns the most common file extension in the list of files.
    """
    extensions = [os.path.splitext(f)[1].lower() for f in files]
    if not extensions:
        return None
    return Counter(extensions).most_common(1)[0][0]

def load_volume_from_dir(in_path):
    """
    Returns npy array with volume by given directory path.
    Automatically detects and uses the most common file extension in the directory.
    Handles different naming conventions (0.ext, 000.ext, etc.)
    """
    # Get all files in directory
    all_files = [f for f in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, f))]
    
    if not all_files:
        raise FileNotFoundError("No files found in the directory")
    
    # Find most common extension
    common_ext = get_most_common_extension(all_files)
    if not common_ext:
        raise ValueError("No files with extensions found in the directory (maybe DICOM).")
    
    print(f"Using files with extension: {common_ext}")
    
    # Filter files by most common extension
    files = [f for f in all_files if os.path.splitext(f)[1].lower() == common_ext]
    
    if not files:
        raise ValueError(f"No files with extension {common_ext} found")
    
    # Natural sort files
    files.sort(key=natural_sort_key)
    
    # Get full paths
    file_paths = [os.path.join(in_path, x) for x in files]
    # Verify first file is a valid image
    try:
        with Image.open(file_paths[0]) as img:
            size_y, size_x = np.array(img).shape
            
    except Exception as e:
        print(f"Error opening image file '{file_paths[0]}': {e}")
        print(f"Note: Files with extension {common_ext} must be valid image files")
        sys.exit(1)
    print(f'size_y, size_x = {size_y, size_x}')
    size_z = len(files)
    return load_volume(file_paths, size_z, size_y, size_x)
