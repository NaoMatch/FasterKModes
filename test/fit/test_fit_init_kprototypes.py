import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from typing import Tuple
from FasterKPrototypes import FasterKPrototypes
from valid_hyperparameters import custom_valid_init_kprototypes


def test_kprototypes_valid_init_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    model = FasterKPrototypes(init=custom_valid_init_kprototypes)
    model.fit(X, categorical=[1,2,3,4])
    assert model.n_clusters == 8

def test_kprototypes_invalid_init_Ccat_type_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return None, Xnum[:n_clusters, :]

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The categorical feature centroids must be a numpy ndarray." in str(excinfo.value)

def test_kprototypes_invalid_init_Ccat_ndim_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return np.random.randint(0, 256, (2,2,2), dtype=np.uint8), Xnum[:n_clusters, :]

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The categorical feature centroids must be a 2-dimensional array." in str(excinfo.value)

def test_kprototypes_invalid_init_Ccat_n_cols_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return np.random.randint(0, 256, (2,5), dtype=np.uint8), Xnum[:n_clusters, :]

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "The categorical feature centroids must have " in str(excinfo.value)

def test_kprototypes_invalid_init_Ccat_dtype_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return np.random.randint(0, 256, (2,4), dtype=np.uint32), Xnum[:n_clusters, :]

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The dtype of categorical feature centroids does not match the expected type derived from X[:, categorical]'s maximum value. " in str(excinfo.value)

def test_kprototypes_invalid_init_Ccat_range_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 8, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return np.random.randint(8, 256, (2,4), dtype=np.uint8), Xnum[:n_clusters, :]

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The categorical feature centroids contain invalid values for categorical features." in str(excinfo.value)



def test_kprototypes_invalid_init_Cnum_type_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return Xcat[:n_clusters, :], None

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The numerical feature centroids must be a numpy ndarray." in str(excinfo.value)

def test_kprototypes_invalid_init_Cnum_ndim_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return Xcat[:n_clusters, :], np.random.randint(0, 256, (2,2,2)).astype(np.float32)

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The numerical feature centroids must be a 2-dimensional array." in str(excinfo.value)

def test_kprototypes_invalid_init_Cnum_n_cols_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return  Xcat[:n_clusters, :], Xnum[:n_clusters, :2]

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The numerical feature centroids must have " in str(excinfo.value)

def test_kprototypes_invalid_init_Cnum_dtype_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return Xcat[:n_clusters, :], Xnum[:n_clusters, :].astype(np.float16)

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The numerical feature centroids must contain float values (np.float32 or np.float64)." in str(excinfo.value)


def test_kprototypes_invalid_init_n_rows_mismatch_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return Xcat[:n_clusters, :], Xnum[:n_clusters+1, :]

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The number of categorical feature centroids and numerical feature centroids must be the same." in str(excinfo.value)

def test_kprototypes_invalid_init_n_clusters_mismatch_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)

    def custom_invalid_init_kprototypes(Xcat: np.array, Xnum: np.array, n_clusters: int) -> Tuple[np.array, np.array]:
        return Xcat[:n_clusters+1, :], Xnum[:n_clusters+1, :]

    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(init=custom_invalid_init_kprototypes)
        model.fit(X, categorical=[1,2,3,4])
    assert "Custom init function output: The number of centroids for categorical and numerical features must match n_clusters." in str(excinfo.value)

