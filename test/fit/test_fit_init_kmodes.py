import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes
from valid_hyperparameters import KMODES_VALID_HYPERPARAMETERS, custom_valid_init_kmodes

def test_kmodes_valid_init_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    model = FasterKModes(init=custom_valid_init_kmodes)
    model.fit(X)
    assert model.n_clusters == 8

def test_kmodes_invalid_init_return_type_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    def custom_invalid_init_return_kmodes(X, n_clusters):
        return None
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(init=custom_invalid_init_return_kmodes)
        model.fit(X)
    assert "self.C must be a numpy ndarray." in str(excinfo.value)

def test_kmodes_invalid_init_ndim_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    def custom_invalid_init_return_kmodes(X, n_clusters):
        return np.random.randint(0, 256, (10, 10, 10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(init=custom_invalid_init_return_kmodes)
        model.fit(X)
    assert "self.C must be a 2-dimensional array." in str(excinfo.value)

def test_kmodes_invalid_init_n_cols_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    def custom_invalid_init_return_kmodes(X, n_clusters):
        return np.random.randint(0, 256, (10, 100), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(init=custom_invalid_init_return_kmodes)
        model.fit(X)
    assert "self.C must have the same number of columns as the categorical features." in str(excinfo.value)

def test_kmodes_invalid_init_dtypes_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    def custom_invalid_init_return_dtypes_kmodes(X, n_clusters):
        return np.random.randint(0, 256, (10, 10), dtype=np.uint32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(init=custom_invalid_init_return_dtypes_kmodes)
        model.fit(X)
    assert "self.C must be np.uint8 or np.uint16, not" in str(excinfo.value)

def test_kmodes_invalid_init_range_fit():
    """有効なデータを渡したときに fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 8, (100, 10), dtype=np.uint8)
    def custom_invalid_init_return_dtypes_kmodes(X, n_clusters):
        return np.random.randint(0, 16, (10, 10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(init=custom_invalid_init_return_dtypes_kmodes)
        model.fit(X)
    assert "init_Ccat contains invalid values for categorical features." in str(excinfo.value)
