import os
import itertools 
import numpy as np
import pandas as pd
import data_loader_utils
from random import shuffle
from pathlib import Path
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from numba import jit

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


def augment(T, X_data, p):
    """
    Augments X_train

    Parameters:
    - T: List of augmentation functions
    - X_data: Array of training examples
    - p: Number of windows

    Returns:
    - X_augmented: Array of augmented training examples

    """
    X = np.copy(X_data)
    X_augmented_list = []
    n_samples, n_timesteps, n_sensors = X.shape

    loc = []
    trans = []
    
    b = compute_windows(n_timesteps, p) # Compute window bounds for p windows

    for i, t in enumerate(T):
        print(f"Processing {t.__name__} transformations")
        for x in X:
            W_j = np.random.randint(1, p + 1)  # Randomly choose a window W_j
            b_lower, b_upper = b[W_j]  # Get the upper and lower bound of the windows
            c1, c2 = sorted(np.random.randint(b_lower, b_upper + 1, size=2))  # Sample two random numbers within the window
            if c2 < c1:
                c1, c2 = c2, c1
            x_augmented = t(x, c1, c2, p, b)  # Augment x in window W_j with t 
            x_scalogram = compute_scalogram(x_augmented, n_sensors)
            X_augmented_list.append(x_scalogram)
            if i == 0:
                loc.append(0)
            else:
                loc.append(W_j)
            
        i += 1
        
    shuffled_indices = np.random.permutation(len(X_augmented_list))
    
    return (np.array(X_augmented_list)[shuffled_indices],
            np.array(loc)[shuffled_indices],
            np.array(trans)[shuffled_indices])


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
    x_aug = np.copy(x)
    return x_aug

def cut_paste(x, c1, c2, p , b):
    """
    Helper function for augment()

    """
    x_aug = np.copy(x)
    cut_snippet = x_aug[c1:c2, :]
    delta = c2 - c1
    
    W_j = np.random.randint(1, p + 1)
    b_lower_j, b_upper_j = b[W_j]

    p1 = np.random.randint(b_lower_j, b_upper_j - delta + 1)
    p2 = p1 + delta

    x_aug[p1:p2, :] = cut_snippet
    return x_aug
 
def mean_shift(x, c1, c2, p = None, b = None):
    """
    Helper function for augment()

    """
    x_aug = np.copy(x)
    time_series_mean = x_aug.mean()
    x_aug[c1:c2] = x_aug[c1:c2] + time_series_mean
    return x_aug


def missing_signal(x, c1, c2, p = None, b = None):
    """
    Helper function for augment()

    """
    x_aug = np.copy(x)
    x_aug[c1:c2] = x_aug[c1]
    return x_aug

from sklearn.preprocessing import MinMaxScaler
from cv2 import resize
from pywt import cwt

def compute_scalogram(x, n_sensors):
    """
    Helper function for augment()

    """
    x_scalogram_list = []
    scaler = MinMaxScaler()
    for m in range(n_sensors):
        cwtmatr, freqs = cwt(x[:, m], np.arange(1, 129), "morl")
        cwtmatr_reshaped = resize(cwtmatr, (128, 512))
        cwtmatr_reshaped_normalized = scaler.fit_transform(cwtmatr_reshaped)
        x_scalogram_list.append(cwtmatr_reshaped_normalized)
    x_scalogram_raw = np.array(x_scalogram_list)
    x_scalogram = x_scalogram_raw.reshape(128, 512, 3)
    return x_scalogram


def shuffle(X, loc, trans):
    p = np.random.permutation(len(X))
    X = np.asarray(X)
    loc = np.asarray(loc)
    trans = np.asarray(trans)
    return X[p], loc[p], trans[p]



def machine_split(df, test_size = 0.7):
    M01 = df[df["MC"] == "M01"]
    X_M01, y_M01 = M01.iloc[:,0:3], M01.iloc[:,-1]
    
    M02 = df[df["MC"] == "M02"]
    X_M02, y_M02 = M02.iloc[:,0:3], M02.iloc[:,-1]
    
    M03 = df[df["MC"] == "M03"]
    X_M03, y_M03 = M03.iloc[:,0:3], M03.iloc[:,-1]
    
    X_M01_train, X_M01_test, y_M01_train, y_M01_test = train_test_split(X_M01, y_M01, test_size = test_size, stratify = y_M01)
    X_M02_train, X_M02_test, y_M02_train, y_M02_test = train_test_split(X_M02, y_M02, test_size = test_size, stratify = y_M02)
        
    X_trainval = pd.concat((X_M01_train, X_M02_train))
    y_trainval = pd.concat((y_M01_train, y_M02_train))
    
    X_test = pd.concat((X_M01_test, X_M02_test, X_M03))
    y_test = pd.concat((y_M01_test, y_M02_test, y_M03))
    return X_trainval, X_test, y_trainval, y_test


def time_split(df, test_size = 0.75):
    Feb_2019 = df[(df["MM"] == "Feb") & (df["YY"] == "2019")]
    Aug_2019 = df[(df["MM"] == "Aug") & (df["YY"] == "2019")]
    Feb_2020 = df[(df["MM"] == "Feb") & (df["YY"] == "2020")]
    Aug_2020 = df[(df["MM"] == "Aug") & (df["YY"] == "2020")]
    Feb_2021 = df[(df["MM"] == "Feb") & (df["YY"] == "2021")]
    Aug_2021 = df[(df["MM"] == "Aug") & (df["YY"] == "2021")]
    
    X_Feb_2019, y_Feb_2019 = Feb_2019.iloc[:,0:3], Feb_2019.iloc[:,-1]
    X_Aug_2019, y_Aug_2019 = Aug_2019.iloc[:,0:3], Aug_2019.iloc[:,-1]
    X_Feb_2020, y_Feb_2020 = Feb_2020.iloc[:,0:3], Feb_2020.iloc[:,-1]
    X_Aug_2020, y_Aug_2020 = Aug_2020.iloc[:,0:3], Aug_2020.iloc[:,-1]
    X_Feb_2021, y_Feb_2021 = Feb_2021.iloc[:,0:3], Feb_2021.iloc[:,-1]
    X_Aug_2021, y_Aug_2021 = Aug_2021.iloc[:,0:3], Aug_2021.iloc[:,-1]
    
    X_Feb_2019_train, X_Feb_2019_test, y_Feb_2019_train, y_Feb_2019_test = train_test_split(X_Feb_2019, y_Feb_2019, test_size = test_size, stratify = y_Feb_2019)
        
    X_Aug_2019_train, X_Aug_2019_test, y_Aug_2019_train, y_Aug_2019_test = train_test_split(X_Aug_2019, y_Aug_2019, test_size = test_size, stratify = y_Aug_2019)
    
    X_Feb_2020_train, X_Feb_2020_test, y_Feb_2020_train, y_Feb_2020_test = train_test_split(X_Feb_2020, y_Feb_2020, test_size = test_size, stratify = y_Feb_2020)
    
    X_Feb_2021_train, X_Feb_2021_test, y_Feb_2021_train, y_Feb_2021_test = train_test_split(X_Feb_2021, y_Feb_2021, test_size = test_size, stratify = y_Feb_2021)
    
    X_trainval = pd.concat((X_Feb_2019_train, X_Aug_2019_train, X_Feb_2020_train, X_Feb_2021_train))
    X_test = pd.concat((X_Feb_2019_test, X_Aug_2019_test, X_Feb_2020_test, X_Aug_2020, X_Feb_2021_test, X_Aug_2021))
    
    y_trainval = pd.concat((y_Feb_2019_train, y_Aug_2019_train, y_Feb_2020_train, y_Feb_2021_train))
    y_test = pd.concat((y_Feb_2019_test, y_Aug_2019_test, y_Feb_2020_test, y_Aug_2020, y_Feb_2021_test, y_Aug_2021))
    return X_trainval, X_test, y_trainval, y_test

def op_split(df, test_size = 0.5):
    OP07 = df[df["OP"] == "OP07"]
    OP01 = df[df["OP"] == "OP01"]
    OP02 = df[df["OP"] == "OP02"]
    OP10 = df[df["OP"] == "OP10"]
    OP04 = df[df["OP"] == "OP04"]
    OP = df[~df["OP"].isin(["OP07", "OP01", "OP02", "OP10", "OP04"])]
    
    X_OP07, y_OP07 = OP07.iloc[:,0:3], OP07.iloc[:,-1]
    X_OP01, y_OP01 = OP01.iloc[:,0:3], OP01.iloc[:,-1]
    X_OP02, y_OP02 = OP02.iloc[:,0:3], OP02.iloc[:,-1]
    X_OP10, y_OP10 = OP10.iloc[:,0:3], OP10.iloc[:,-1]
    X_OP04, y_OP04 = OP04.iloc[:,0:3], OP04.iloc[:,-1]
    X_OP, y_OP = OP.iloc[:,0:3], OP.iloc[:,-1]
    
    
    X_OP07_train, X_OP07_test, y_OP07_train, y_OP07_test = train_test_split(X_OP07, y_OP07, test_size = test_size, stratify = y_OP07)
    
    X_OP01_train, X_OP01_test, y_OP01_train, y_OP01_test = train_test_split(X_OP01, y_OP01, test_size = test_size, stratify = y_OP01)
    
    X_OP02_train, X_OP02_test, y_OP02_train, y_OP02_test = train_test_split(X_OP02, y_OP02, test_size = test_size, stratify = y_OP02)
    
    X_OP10_train, X_OP10_test, y_OP10_train, y_OP10_test = train_test_split(X_OP10, y_OP10, test_size = test_size, stratify = y_OP10)
    
    X_OP04_train, X_OP04_test, y_OP04_train, y_OP04_test = train_test_split(X_OP04, y_OP04, test_size = test_size, stratify = y_OP04)
    
    X_trainval = pd.concat((X_OP07_train, X_OP01_train, X_OP02_train, X_OP10_train, X_OP04_train))
    X_test = pd.concat((X_OP07_test, X_OP01_test, X_OP02_test, X_OP10_test, X_OP04_test, X_OP))
    
    y_trainval = pd.concat((y_OP07_train, y_OP01_train, y_OP02_train, y_OP10_train, y_OP04_train))
    y_test = pd.concat((y_OP07_test, y_OP01_test, y_OP02_test, y_OP10_test, y_OP04_test, y_OP))
    return X_trainval, X_test, y_trainval, y_test