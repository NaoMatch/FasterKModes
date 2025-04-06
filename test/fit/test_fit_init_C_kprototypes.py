"""
FasterKPrototypesクラスタリングアルゴリズムのfitメソッドに対する
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
    """fitが数値とカテゴリの混在データで正常動作するかを検証"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
    model.fit(X, categorical=[1,3,5,7])
    assert model.n_clusters == 2

def test_kprototypes_valid_init_C_fit():
    """正しい初期セントロイド(init_C)指定時のfit動作を検証"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:2, :4]
    init_Cnum = X[2:4,4:].astype(np.float32)
    model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
    model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert model.n_clusters == 2

def test_kprototypes_invalid_init_C_either_none_fit():
    """init_Ccatまたはinit_Cnumの一方のみ指定時にValueErrorを確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:2, :4]
    init_Cnum = None
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Ccat and init_Cnum must either both be None or both be provided." in str(excinfo.value)

    init_Ccat = None
    init_Cnum = X[2:4,4:].astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Ccat and init_Cnum must either both be None or both be provided." in str(excinfo.value)

def test_kprototypes_invalid_init_Ccat_type_fit():
    """init_Ccatにlistを指定した際にValueErrorを確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = [
        [1,2], 
        [3,4]
    ]
    init_Cnum = X[2:4,4:].astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Ccat must be a numpy ndarray." in str(excinfo.value)

def test_kprototypes_valid_init_Ccat_ndim_fit():
    """init_Ccatが3次元配列の際にValueErrorを確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = np.random.randint(0, 256, (2, 10, 2), dtype=np.uint8)
    init_Cnum = X[2:4,4:].astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Ccat must be a 2-dimensional array." in str(excinfo.value)

def test_kprototypes_invalid_init_Ccat_shape_fit():
    """init_Ccatの列数とカテゴリ変数数の不一致でValueError確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:2, :4]
    init_Cnum = X[2:4,4:].astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7,9], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Ccat must have the same number of columns as the categorical features." in str(excinfo.value)

def test_kprototypes_invalid_init_Ccat_type_fit():
    """init_Ccatのdtypeが想定外の場合にValueError確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:2, :4].astype(np.uint32)
    init_Cnum = X[2:4,4:].astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "The dtype of init_Ccat does not match the expected type derived " in str(excinfo.value)

def test_kprototypes_invalid_init_Cnum_type_fit():
    """init_Cnumにlistを指定した際にValueErrorを確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:2, :4].astype(np.uint8)
    init_Cnum = [
        [1,2], 
        [3,4]
    ]
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Cnum must be a numpy ndarray." in str(excinfo.value)

def test_kprototypes_invalid_init_Cnum_ndim_fit():
    """init_Cnumが3次元配列の際にValueErrorを確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:2, :4].astype(np.uint8)
    init_Cnum = np.random.randint(0, 256, (2, 10, 10)).astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Cnum must be a 2-dimensional array." in str(excinfo.value)

def test_kprototypes_invalid_init_Cnum_shape_fit():
    """init_Cnumの列数が数値変数数と異なる場合にValueError確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:2, :4].astype(np.uint8)
    init_Cnum = X[2:4,5:].astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Cnum must have the same number of columns as the numerical features." in str(excinfo.value)

def test_kprototypes_invalid_init_Cnum_dtype_fit():
    """init_Cnumのdtypeがfloat16の場合にValueErrorを確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:2, :4].astype(np.uint8)
    init_Cnum = X[2:4,4:].astype(np.float16)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "The dtype of init_Cnum does not match the expected type derived from the maximum value " in str(excinfo.value)

def test_kprototypes_invalid_init_C_n_rows_mismatch_fit():
    """init_Ccatとinit_Cnumの行数不一致時にValueErrorを確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:2, :4].astype(np.uint8)
    init_Cnum = X[2:5,4:].astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Ccat and init_Cnum must have the same number of rows (centroids)." in str(excinfo.value)

def test_kprototypes_invalid_init_C_n_clusters_mismatch_fit():
    """初期クラスタ数とn_clustersの不一致時にValueErrorを確認"""
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    init_Ccat = X[:3, :4].astype(np.uint8)
    init_Cnum = X[2:5,4:].astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,3,5,7], init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "The number of rows in init_Ccat and init_Cnum must match n_clusters." in str(excinfo.value)
