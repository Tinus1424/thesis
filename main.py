import os
import itertools 
import numpy as np
import pandas as pd
import data_loader_utils
from pathlib import Path
import matplotlib.pyplot as plt 

def preprocess_data(X_data, y_data):
    """
    Preprocesses data and splits y into target and additional features

    Parameters:
    - X_data: List of 3D arrays
    - y_data: List of labels

    Returns:
    - X: List of 3D arrays in np.float32
    - y: List of target labels
    - features: List of additional labels
    """
    y_split = [y.split("_") for y in y_data] # Split the Machine, Month, Year, Process, ExampleId, Target
    y_np = np.array(y_split) # Convert to np for easier indexing
    ytarget = y_np[:, -1] # Extract target value
    features = list(y_np[:,:-1]) # Extract additional information
    y = [1 if "good" in y else 0 for y in ytarget] # List of y values
    X = [X.astype(np.float32) for X in X_data] # List of X values in the same dtype
    return X, y, features


def window_data(X, y, features, window_size):
    """
    Windows the data given a window_size:

    Parameters:
    - X: List of 3D arrays
    - y: List of target values
    - features: List of additional features
    - window_size: window size

    Returns:
    - npX: 3D numpy array
    - npy: 1D numpy array
    - npf: 2D Numpy array
    """
    npX = np.empty((0, 3, window_size))
    npy = np.empty((0, ))
    npf = np.empty((0, 5))
    for example in zip(X, y, features):
        
        modulo = example[0].shape[0] % window_size
        floor = example[0].shape[0] // window_size
        
        if modulo > 0: 
            appendX = example[0][:-modulo,:].copy()
        else:
            appendX = example[0].copy()
            
        appendX = np.reshape(appendX, (floor, 3, window_size))
        npX = np.concatenate((npX, appendX))
        
        if example[1] == 0:
            npy = np.concatenate((npy, np.zeros(floor)))
        else:
            npy = np.concatenate((npy, np.ones(floor)))

        appendf = np.array(example[2])
        appendf = np.tile(appendf, (floor, 1))

        npf = np.concatenate((npf, appendf))
        
    return npX, npy, npf