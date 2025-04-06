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
from FasterKModes import FasterKPrototypes
from valid_hyperparameters import KMODES_VALID_HYPERPARAMETERS, custom_valid_init_kmodes

def test_kprototypes_valid_fit():
    """数値データに対してfitが正常に動作するか検証する"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
    model.fit(X, categorical=[1,3,5,7])
    assert model.n_clusters == 2

def test_kprototypes_valid_fit():
    """浮動小数点データに対してfitが正常に動作するか検証する"""
    X = np.random.uniform(0, 256, (100, 10))
    model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
    model.fit(X, categorical=[1,3,5,7])
    assert model.n_clusters == 2

def test_kprototypes_invalid_nan_fit():
    """NaNを含むデータを渡したときにエラーが出るか検証する"""
    X = np.random.uniform(0, 256, (100, 10))
    X[5,5] = np.nan
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7])
    assert "X must not contain NaN values." in str(excinfo.value)

def test_kprototypes_invalid_categorical_null_fit():
    """categoricalが空のときにエラーが出るか検証する"""
    X = np.random.uniform(0, 256, (100, 10))
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[])
    assert "No categorical features provided. Using sklearn.cluster.KMeans." in str(excinfo.value)

def test_kprototypes_invalid_categorical_all_fit():
    """全てcategoricalなときにエラーが出るか検証する"""
    X = np.random.uniform(0, 256, (100, 10))
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[0,1,2,3,4,5,6,7,8,9])
    assert "All features are categorical. Using FasterKModes." in str(excinfo.value)

def test_kprototypes_invalid_categorical_type_fit():
    """categoricalにリスト以外を渡したときエラーになるか検証"""
    X = np.random.uniform(0, 256, (100, 10))
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=1.0)
    assert "'categorical' must be a list." in str(excinfo.value)

def test_kprototypes_invalid_categorical_contain_non_int_fit():
    """categoricalに整数以外が含まれるときエラーか検証する"""
    X = np.random.uniform(0, 256, (100, 10))
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[0, 1, 2.0])
    assert "All elements in 'categorical' must be integers." in str(excinfo.value)

def test_kprototypes_invalid_categorical_duplicate_fit():
    """categoricalに重複があるときエラーが出るか検証する"""
    X = np.random.uniform(0, 256, (100, 10))
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[0, 1, 1])
    assert "All elements in 'categorical' must be unique." in str(excinfo.value)

def test_kprototypes_invalid_without_categorical_fit():
    """categoricalを省略したときにTypeErrorが出るか検証する"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    with pytest.raises(TypeError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
    assert "FasterKPrototypes.fit() missing 1 required positional argument: 'categorical'" in str(excinfo.value)

def test_kprototypes_invalid_input_type_fit():
    """ndarray以外の入力でfitが失敗するかを検証する"""
    X = [
        [1,2,3,4,5], 
        [2,3,4,5,6], 
    ]
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7])
    assert "Error: X must be a numpy array. Current input type:" in str(excinfo.value)

def test_kprototypes_invalid_input_numpy_type_fit():
    """不適切なdtypeを持つndarrayに対するエラーを検証する"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint64)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7])
    assert "X must have a dtype of uint8, uint16, float32, or float64." in str(excinfo.value)

def test_kprototypes_invalid_input_ndim_fit():
    """2次元以外の配列を渡すとエラーになるか検証する"""
    X = np.random.randint(0, 256, (100, 10, 10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7])
    assert "Error: X must be a 2D array. Current ndim: " in str(excinfo.value)

def test_kprototypes_invalid_input_dtype_fit():
    """許可されていないdtypeに対してエラーが出るか検証する"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7])
    assert "X must have a dtype of uint8, uint16, float32, or float64." in str(excinfo.value)
