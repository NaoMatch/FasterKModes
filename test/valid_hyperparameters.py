import typing
from typing import Tuple
import numpy as np

# init
def custom_valid_init_kmodes(X: np.array, n_clusters: int) -> np.array:
    return X[:n_clusters,:]

def custom_valid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
    return Xcat[:n_clusters, :], Xnum[:n_clusters, :]

def custom_invalid_args_init_kmodes(Xcat: np.array, n_clusters: int) -> np.array:
    return Xcat[:n_clusters,:]

def custom_invalid_args_init_kprototypes(Xcat: np.array, n_clusters: int) -> np.array:
    return Xcat[:n_clusters,:]


# categorical_measure
def custom_valid_args_categorical_measure(x_cat, c_cat):
    return np.sum(x_cat != c_cat).sum()

def custom_invalid_args_categorical_measure(x, c):
    return np.sum(x != c).sum()

# numerical_measure
def custom_valid_args_numerical_measure(x_num, c_num):
    return np.linalg.norm(x_num-c_num)

def custom_invalid_args_numerical_measure(x, c):
    return np.linalg.norm(x-c)

KMODES_VALID_HYPERPARAMETERS = {
    "n_clusters":2,
    "max_iter":1, 
    "min_n_moves":0,
    "n_init":1,
    "random_state":42,
    "init":"random",
    "categorical_measure": "hamming",
    "n_jobs":1,
    "print_log":False,
    "recompile":False,
    "use_simd":False,
    "max_tol":0.1        
}

KPROTOTYPES_VALID_HYPERPARAMETERS = {
    "n_clusters":2,
    "max_iter":1, 
    "min_n_moves":0,
    "n_init":1,
    "random_state":42,
    "init":"random",
    "categorical_measure": "hamming",
    "numerical_measure": "euclidean",
    "n_jobs":1,
    "gamma":1.0,
    "print_log":False,
    "recompile":False,
    "use_simd":False,
    "max_tol":0.1        
}


