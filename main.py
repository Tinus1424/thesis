import os
import itertools 
import numpy as np
import pandas as pd
import data_loader_utils
from pathlib import Path

import scipy.stats as stats
import statsmodels.api as sm

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



def qqplot(sample, distributions):
    """
    Draws Q-Q plots for a sample given distributions and degrees of freedom

    Parameters:
    - sample: 2D arrray
    - distributions: Dictionary with names for keys and stats distributions for values
    """

    axis = ["X", "Y", "Z"]
    for title, dist in distributions.items():
        fig, ax = plt.subplots(1, 3, figsize = (20, 4))
        for i in range(len(axis)):
            sm.qqplot(sample[:, i], dist, fit = True, line = "s", ax = ax[i])
            ax[i].set_title(f"{axis[i]}-axis")
        fig.suptitle(f"Q-Q plot for {title} distribution")
    return
