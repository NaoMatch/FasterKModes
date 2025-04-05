"""
FasterKModesクラスタリングアルゴリズムのfitメソッドに対する
様々な入力バリデーションのユニットテストを定義している
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes
from valid_hyperparameters import KMODES_VALID_HYPERPARAMETERS, custom_valid_init_kmodes

def test_kmodes_valid_fit():
    """有効な入力データを用いてfitが正常に実行されるか確認する"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
    model.fit(X)
    assert model.n_clusters == 2

def test_kmodes_invalid_input_type_fit():
    """fitにndarray以外を渡した際にValueErrorが発生するか確認する"""
    X = [
        [1,2,3,4,5], 
        [2,3,4,5,6], 
    ]
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
    assert "Error: X must be a numpy array. Current input type" in str(excinfo.value)

def test_kmodes_invalid_input_ndim_fit():
    """fitに2次元以外の配列を渡した際にValueErrorが発生するか確認する"""
    X = np.random.randint(0, 256, (100, 10, 10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
    assert "Error: X must be a 2D array. Current ndim" in str(excinfo.value)

def test_kmodes_invalid_input_dtype_fit():
    """fitに許可されていないdtypeを渡した際にValueErrorが発生するか確認する"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
    assert "Error: X's dtype must be uint8 or uint16. Current dtype" in str(excinfo.value)

def test_kmodes_invalid_input_order_fit():
    """fitにメモリ順序がCでない配列を渡した際にValueErrorが発生するか確認する"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint16)
    X = np.array(X, order="F")
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
    assert "X must have order='C'." in str(excinfo.value)

def test_kmodes_invalid_input_n_unique_rows_fit():
    """fitにユニークな行が足りない配列を渡した際にValueErrorが発生するか確認する"""
    X = np.random.randint(0, 1, (100, 10), dtype=np.uint16)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
    assert "The number of unique rows in X" in str(excinfo.value)
