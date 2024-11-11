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