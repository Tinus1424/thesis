import os
import itertools 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle
import data_loader_utils
from random import shuffle
from pathlib import Path


from sklearn.metrics import f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, ParameterGrid




def get_df(X_data, y_data):
    X = np.array([x.astype(np.float64)[:4096, :3] for x in X_data])
    y = np.array([0 if id.split("_")[-1] == "good" else 1 for id in y_data])
    
    axis = ["X-axis", "Y-axis", "Z-axis"] 
    axisdict = {"X-axis": [], "Y-axis":[],  "Z-axis":[]}
    for i, ax in enumerate(axis):
        for n in range(X.shape[0]):
            axisdict[ax].append(pd.Series(X[n][:,i]))
    
    X_df = pd.DataFrame(axisdict)
    
    y_df = pd.DataFrame([y.split("_") for y in y_data])
    df = X_df.join(y_df).rename(columns = {0: "MC", 1: "MM", 2: "YY", 3: "OP", 4: "n", 5: "y"})
    df["y"] = df["y"].apply(lambda x: 1 if x == "bad" else 0)
    return df


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

from sklearn.model_selection import train_test_split
def machine_split(df, m = 3, test_size = 0.7):
    seed = 27
    M01 = df[df["MC"] == "M01"]
    X_M01, y_M01 = M01.iloc[:,:m], M01.iloc[:,-1]
    
    M02 = df[df["MC"] == "M02"]
    X_M02, y_M02 = M02.iloc[:,:m], M02.iloc[:,-1]
    
    M03 = df[df["MC"] == "M03"]
    X_M03, y_M03 = M03.iloc[:,:m], M03.iloc[:,-1]
    
    X_M01_train, X_M01_test, y_M01_train, y_M01_test = train_test_split(X_M01, y_M01, test_size = test_size, stratify = y_M01, random_state = seed)
    X_M02_train, X_M02_test, y_M02_train, y_M02_test = train_test_split(X_M02, y_M02, test_size = test_size, stratify = y_M02, random_state = seed)
        
    X_trainval = pd.concat((X_M01_train, X_M02_train))
    y_trainval = pd.concat((y_M01_train, y_M02_train))
    
    X_test = pd.concat((X_M01_test, X_M02_test, X_M03))
    y_test = pd.concat((y_M01_test, y_M02_test, y_M03))
    return X_trainval, X_test, y_trainval, y_test


def time_split(df, m = 3, test_size = 0.75):
    seed = 27
    Feb_2019 = df[(df["MM"] == "Feb") & (df["YY"] == "2019")]
    Aug_2019 = df[(df["MM"] == "Aug") & (df["YY"] == "2019")]
    Feb_2020 = df[(df["MM"] == "Feb") & (df["YY"] == "2020")]
    Aug_2020 = df[(df["MM"] == "Aug") & (df["YY"] == "2020")]
    Feb_2021 = df[(df["MM"] == "Feb") & (df["YY"] == "2021")]
    Aug_2021 = df[(df["MM"] == "Aug") & (df["YY"] == "2021")]
    
    X_Feb_2019, y_Feb_2019 = Feb_2019.iloc[:,:m], Feb_2019.iloc[:,-1]
    X_Aug_2019, y_Aug_2019 = Aug_2019.iloc[:,:m], Aug_2019.iloc[:,-1]
    X_Feb_2020, y_Feb_2020 = Feb_2020.iloc[:,:m], Feb_2020.iloc[:,-1]
    X_Aug_2020, y_Aug_2020 = Aug_2020.iloc[:,:m], Aug_2020.iloc[:,-1]
    X_Feb_2021, y_Feb_2021 = Feb_2021.iloc[:,:m], Feb_2021.iloc[:,-1]
    X_Aug_2021, y_Aug_2021 = Aug_2021.iloc[:,:m], Aug_2021.iloc[:,-1]
    
    X_Feb_2019_train, X_Feb_2019_test, y_Feb_2019_train, y_Feb_2019_test = train_test_split(X_Feb_2019, y_Feb_2019, test_size = test_size, stratify = y_Feb_2019, random_state = seed)
        
    X_Aug_2019_train, X_Aug_2019_test, y_Aug_2019_train, y_Aug_2019_test = train_test_split(X_Aug_2019, y_Aug_2019, test_size = test_size, stratify = y_Aug_2019, random_state = seed)
    
    X_Feb_2020_train, X_Feb_2020_test, y_Feb_2020_train, y_Feb_2020_test = train_test_split(X_Feb_2020, y_Feb_2020, test_size = test_size, stratify = y_Feb_2020, random_state = seed)
    
    X_Feb_2021_train, X_Feb_2021_test, y_Feb_2021_train, y_Feb_2021_test = train_test_split(X_Feb_2021, y_Feb_2021, test_size = test_size, stratify = y_Feb_2021, random_state = seed)
    
    X_trainval = pd.concat((X_Feb_2019_train, X_Aug_2019_train, X_Feb_2020_train, X_Feb_2021_train))
    X_test = pd.concat((X_Feb_2019_test, X_Aug_2019_test, X_Feb_2020_test, X_Aug_2020, X_Feb_2021_test, X_Aug_2021))
    
    y_trainval = pd.concat((y_Feb_2019_train, y_Aug_2019_train, y_Feb_2020_train, y_Feb_2021_train))
    y_test = pd.concat((y_Feb_2019_test, y_Aug_2019_test, y_Feb_2020_test, y_Aug_2020, y_Feb_2021_test, y_Aug_2021))
    return X_trainval, X_test, y_trainval, y_test

def op_split(df, m = 3, test_size = 0.5):
    seed = 27
    OP07 = df[df["OP"] == "OP07"]
    OP01 = df[df["OP"] == "OP01"]
    OP02 = df[df["OP"] == "OP02"]
    OP10 = df[df["OP"] == "OP10"]
    OP04 = df[df["OP"] == "OP04"]
    OP = df[~df["OP"].isin(["OP07", "OP01", "OP02", "OP10", "OP04"])]
    
    X_OP07, y_OP07 = OP07.iloc[:,:m], OP07.iloc[:,-1]
    X_OP01, y_OP01 = OP01.iloc[:,:m], OP01.iloc[:,-1]
    X_OP02, y_OP02 = OP02.iloc[:,:m], OP02.iloc[:,-1]
    X_OP10, y_OP10 = OP10.iloc[:,:m], OP10.iloc[:,-1]
    X_OP04, y_OP04 = OP04.iloc[:,:m], OP04.iloc[:,-1]
    X_OP, y_OP = OP.iloc[:,:m], OP.iloc[:,-1]
    
    
    X_OP07_train, X_OP07_test, y_OP07_train, y_OP07_test = train_test_split(X_OP07, y_OP07, test_size = test_size, stratify = y_OP07, random_state = seed)
    
    X_OP01_train, X_OP01_test, y_OP01_train, y_OP01_test = train_test_split(X_OP01, y_OP01, test_size = test_size, stratify = y_OP01, random_state = seed)
    
    X_OP02_train, X_OP02_test, y_OP02_train, y_OP02_test = train_test_split(X_OP02, y_OP02, test_size = test_size, stratify = y_OP02, random_state = seed)
    
    X_OP10_train, X_OP10_test, y_OP10_train, y_OP10_test = train_test_split(X_OP10, y_OP10, test_size = test_size, stratify = y_OP10, random_state = seed)
    
    X_OP04_train, X_OP04_test, y_OP04_train, y_OP04_test = train_test_split(X_OP04, y_OP04, test_size = test_size, stratify = y_OP04, random_state = seed)
    
    X_trainval = pd.concat((X_OP07_train, X_OP01_train, X_OP02_train, X_OP10_train, X_OP04_train))
    X_test = pd.concat((X_OP07_test, X_OP01_test, X_OP02_test, X_OP10_test, X_OP04_test, X_OP))
    
    y_trainval = pd.concat((y_OP07_train, y_OP01_train, y_OP02_train, y_OP10_train, y_OP04_train))
    y_test = pd.concat((y_OP07_test, y_OP01_test, y_OP02_test, y_OP10_test, y_OP04_test, y_OP))
    return X_trainval, X_test, y_trainval, y_test


from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
split_functions = [machine_split, time_split, op_split] 
splits = ["machine", "time", "operation"]
def get_uni_cv_results(clf, param_grid, df, n_jobs = -1, splits = splits, split_functions = split_functions):
    cv_results = {}
    gs_objects = {}
    for i, split in enumerate(splits):
        print(f"Hyperparameter tuning on{split}-wise split...")
        splitter = split_functions[i]
        X_train, __, y_train, __ = splitter(df)
        X_train = from_nested_to_3d_numpy(X_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        gs = GridSearchCV(clf, 
                          param_grid, 
                          scoring = "f1", 
                          n_jobs = n_jobs, 
                          cv = StratifiedKFold(n_splits = 3)
                         )
        gs.fit(X_train, y_train)

        cv_results[split] = gs.cv_results_
        gs_objects[split] = gs
    return cv_results, gs_objects


def get_cv_results(clf, param_grid, df, n_jobs = -1, splits = splits, split_functions = split_functions):
    cv_results = {}
    gs_objects = {}
    for i, split in enumerate(splits):
        print(f"Hyperparameter tuning on {split}-wise split...")
        splitter = split_functions[i]
        X_train, __, y_train, __ = splitter(df)
        gs = GridSearchCV(clf, 
                          param_grid, 
                          scoring = "f1", 
                          n_jobs = n_jobs, 
                          cv = StratifiedKFold(n_splits = 3)
                         )
        gs.fit(X_train, y_train)

        cv_results[split] = gs.cv_results_
        gs_objects[split] = gs
    return cv_results, gs_objects


def extract_mean_rank(cv_results):
    n_models = pd.DataFrame(cv_results["machine"]).shape[0]
    mean_rank = {}
    
    for params in range(n_models):
        total_rank = 0
        for split in splits:
            split_df = pd.DataFrame(cv_results[split])
            total_rank += split_df.iloc[params]["rank_test_score"]
        mean_rank[params] = total_rank / len(splits)
    best_model = min(mean_rank, key = mean_rank.get)
    best_params = split_df.iloc[best_model]["params"]
    return mean_rank, best_params

def get_uni_test_results(model, df):
    model_f1 = []
    model_recall = []
    model_cm = []
    model_objects = []
    
    for i, split in enumerate(splits):
        print(f"Testing {split}-wise split")
        splitter = split_functions[i]
        
        X_train, X_test, y_train, y_test = splitter(df, m = 1)
        X_train = from_nested_to_3d_numpy(X_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_test = from_nested_to_3d_numpy(X_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        model_f1.append(f1_score(y_test, y_preds))
        model_recall.append(recall_score(y_test, y_preds))
        model_cm.append(confusion_matrix(y_test, y_preds))
        model_objects.append(model)
    
    results_model = {"model_f1": model_f1, "model_recall": model_recall, "model_cm": model_cm}
    return results_model, model_objects

def get_test_results(model, df):
    model_f1 = []
    model_recall = []
    model_cm = []
    model_objects = []
    
    for i, split in enumerate(splits):
        print(f"Testing {split}-wise split")
        splitter = split_functions[i]
        
        X_train, X_test, y_train, y_test = splitter(df)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        
        model_f1.append(f1_score(y_test, y_preds))
        model_recall.append(recall_score(y_test, y_preds))
        model_cm.append(confusion_matrix(y_test, y_preds))
        model_objects.append(model)
    
    results_model = {"model_f1": model_f1, "model_recall": model_recall, "model_cm": model_cm}
    return results_model, model_objects


