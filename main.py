import os
import itertools 
import numpy as np
import pandas as pd
import data_loader_utils
from random import shuffle
from pathlib import Path
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

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