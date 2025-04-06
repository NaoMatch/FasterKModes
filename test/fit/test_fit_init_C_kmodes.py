import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKModes import FasterKModes
from valid_hyperparameters import KMODES_VALID_HYPERPARAMETERS, custom_valid_init_kmodes

def test_kmodes_valid_init_C_fit():
    """有効な init_C を指定した場合に fit が正常に動作するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_C = X[:KMODES_VALID_HYPERPARAMETERS["n_clusters"],:]
    model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
    model.fit(X, init_C=init_C)
    assert model.n_clusters == 2

def test_kmodes_incvalid_init_C_init_fit():
    """
    numpy 配列以外を init_C に渡すと ValueError が発生するかを
    テストする
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_C = [
        [1,2], 
        [3,4]
    ]
    init = custom_valid_init_kmodes
    with pytest.raises(ValueError) as excinfo:
        from copy import deepcopy
        params = deepcopy(KMODES_VALID_HYPERPARAMETERS)
        params["init"] = init
        model = FasterKModes(**params)
        model.fit(X, init_C=init_C)
    assert "Cannot provide both a custom init function" in str(excinfo.value)

def test_kmodes_incvalid_init_C_type_fit():
    """init_C に numpy 配列以外を渡した場合に ValueError が発生するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_C = [
        [1,2], 
        [3,4]
    ]
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, init_C=init_C)
    assert "Error: init_C must be a numpy array. Current input type: " in str(excinfo.value)

def test_kmodes_incvalid_init_C_ndim_fit():
    """init_C に 2 次元以外の配列を渡した場合に ValueError が発生するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_C = np.random.randint(0, 256, (KMODES_VALID_HYPERPARAMETERS["n_clusters"], 10, 10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, init_C=init_C)
    assert "Error: init_C must be a 2D array. Current ndim: " in str(excinfo.value)

def test_kmodes_incvalid_init_C_n_cols_fit():
    """init_C に 2 次元以外の配列を渡した場合に ValueError が発生するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_C = np.random.randint(0, 256, (KMODES_VALID_HYPERPARAMETERS["n_clusters"], 100), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, init_C=init_C)
    assert "Error: init_C must have the same number of columns as X. " in str(excinfo.value)

def test_kmodes_incvalid_init_C_dtypes_fit():
    """init_C に 2 次元以外の配列を渡した場合に ValueError が発生するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_C = np.random.randint(0, 256, (KMODES_VALID_HYPERPARAMETERS["n_clusters"], 10), dtype=np.uint16)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, init_C=init_C)
    assert "Error: init_C's dtype must match X's dtype. " in str(excinfo.value)

def test_kmodes_incvalid_init_C_outside_fit():
    """init_C に 2 次元以外の配列を渡した場合に ValueError が発生するかをテストする。"""
    X = np.random.randint(0, 8, (100, 10), dtype=np.uint8)
    init_C = np.random.randint(0, 16, (KMODES_VALID_HYPERPARAMETERS["n_clusters"], 10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, init_C=init_C)
    assert "Error: init_C contains values outside the range of X." in str(excinfo.value)

def test_kmodes_incvalid_init_C_n_clusters_fit():
    """init_C に 2 次元以外の配列を渡した場合に ValueError が発生するかをテストする。"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_C = np.random.randint(0, 256, (KMODES_VALID_HYPERPARAMETERS["n_clusters"]+1, 10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, init_C=init_C)
    assert "Error: init_C must have the same number of rows as the number of clusters (n_clusters). " in str(excinfo.value)
