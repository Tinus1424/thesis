import os
import itertools 
import numpy as np
import pandas as pd
import data_loader_utils
from pathlib import Path
import matplotlib.pyplot as plt 

from cv2 import resize
from pywt import cwt



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

def plot_class_dist(y, features):
    """
    Plots the class distribution operation wise

    Parameters: 
    - y: 1D array of binary values
    - features: 2D array with additional features

    Returns:
    Class distribution plot
    """
    y_reshaped = y.reshape((-1, 1))
    plots = np.hstack((features, y_reshaped))
    
    for machine in np.unique(features[:, 0]):
        indices = np.where(plots[:, 0] == machine)
        machineplots = plots[indices]
        counts = {operation: {'0s': 0, '1s': 0} for operation in np.unique(machineplots[:, 3])}
        
        for value, operation in zip(machineplots[:, -1], machineplots[:, 3]):
            if value == "1":
                counts[operation]['1s'] += 1
            else:
                counts[operation]['0s'] += 1
        
        labels = list(counts.keys())
        count_zeros = [counts[label]['0s'] for label in labels]
        count_ones = [counts[label]['1s'] for label in labels]
        
        fig, ax = plt.subplots(figsize=(20, 5))
        
        bars1 = ax.bar(labels, count_zeros, label="Abnormal", color="red")
        bars2 = ax.bar(labels, count_ones, bottom=count_zeros, label="Normal", color="blue")  # Stack on top of 0s
        ax.set_xlim(-0.5, 15-0.5)
        
        ax.set_xlabel("Operations")
        ax.set_ylabel("Counts")
        ax.set_title(f'Counts of Abnormal and Normal Processes for {machine}')
        ax.legend()
        plt.show()


def augment(T, X_train, p):
    """
    Augments X_train

    Parameters:
    - T: List of augmentation functions
    - X_train: Array of training examples
    - p: Number of windows

    Returns:
    - X_augmented_arr: Array of augmented training examples

    """
    X = np.copy(X_train)
    X_augmented_list = []
    n_samples, n_timesteps, n_sensors = X.shape

    b = compute_windows(n_timesteps, p) # Compute window bounds for p windows

    for t in T:
        print(f"Processing {t} transformations")
        for x in X:
            W_j = np.random.randint(1, p + 1) # Randomly choose a window W_j

            b_lower, b_upper = b[W_j] # Get the upper and lower bound of the windows
            c1 = np.random.randint(b_lower, b_upper + 1) # Sample two random numbers within the window
            c2 = np.random.randint(b_lower, b_upper + 1)

            if c1 > c2: # Ensure c1 > c2
                c1, c2 = c2, c1
                
            x_augmented = t(x, c1, c2, p, b) # Augment x in window W_j with Ti 
            x_scalogram = compute_scalogram(x_augmented, n_sensors)
            X_augmented_list.append(x_scalogram)

    X_augmented = np.array(X_augmented_list)
    return X_augmented_arr


def compute_windows(n_timesteps, p):
    """
    Helper function for augment()

    """
    windows = {}
    
    window_size = n_timesteps // p
    remainder = n_timesteps % p
    
    start = 0
    for i in range(1, p +1):
        end = start + window_size + (1 if i <= remainder else 0)
        windows[i] = (start, end -1)
        start = end
        
    return windows

def identity(x, c1, c2, p = None, b = None):
    """
    Helper function for augment()

    """
    return x

def cut_paste(x, c1, c2, p , b):
    """
    Helper function for augment()

    """
    cut_snippet = x[c1:c2, :]
    delta = c2 - c1
    
    W_j = np.random.randint(1, p + 1)
    b_lower_j, b_upper_j = b[W_j]

    p1 = np.random.randint(b_lower_j, b_upper_j - delta + 1)
    p2 = p1 + delta

    x[p1:p2, :] = cut_snippet
    return x
    
def mean_shift(x, c1, c2, p = None, b = None):
    """
    Helper function for augment()

    """
    time_series_mean = x[c1:c2].mean(axis=0)
    x[c1:c2] = x[c1:c2] + time_series_mean
    return x


def missing_signal(x, c1, c2, p = None, b = None):
    """
    Helper function for augment()

    """
    constant = x[c1].copy()
    x[c1:c2] = constant
    return x

def compute_scalogram(x, n_sensors):
    """
    Helper function for augment()

    """
    x_scalogram_list = []
    
    for m in range(n_sensors):
        cwtmatr, freqs = cwt(x[:, m], np.arange(1, 129), "morl")
        cwtmatr_reshaped = resize(cwtmatr, (128, 512))
        x_scalogram_list.append(cwtmatr_reshaped)
        
    x_scalogram_raw = np.array(x_scalogram_list)
    x_scalogram = x_scalogram_raw.reshape(128, 512, 3)
    return x_scalogram



