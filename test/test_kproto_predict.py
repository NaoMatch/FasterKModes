import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from FasterKPrototypes import FasterKPrototypes

# --- ヘルパー関数 ---
def create_valid_X(n_samples=5, n_features=4, dtype=np.uint8, order="C"):
    """
    X は 5x4 の行列で、最初の2列をカテゴリカル（例：1～5の整数）、
    残り2列を数値特徴として扱う。
    """
    X = np.array([
        [1, 2, 10, 20],
        [2, 3, 11, 21],
        [3, 4, 12, 22],
        [4, 5, 13, 23],
        [5, 1, 14, 24]
    ], dtype=dtype, order=order)
    return X

def create_valid_init_Ccat(n_clusters=3, n_cat_features=2, dtype=np.uint8):
    """
    init_Ccat は (n_clusters, n_cat_features) の配列。
    X のカテゴリカル部分（例：1～5の値）の範囲内で設定。
    """
    init_Ccat = np.array([
        [1, 2],
        [2, 3],
        [3, 4]
    ], dtype=dtype, order='C')
    return init_Ccat

def create_valid_init_Cnum(n_clusters=3, n_num_features=2, dtype=np.float32):
    """
    init_Cnum は (n_clusters, n_num_features) の配列。
    数値特徴（例：10～24の値）に合わせた float 型の値を設定。
    """
    init_Cnum = np.array([
        [10.0, 20.0],
        [11.0, 21.0],
        [12.0, 22.0]
    ], dtype=dtype, order='C')
    return init_Cnum

# 共通のハイパーパラメータ（FasterKPrototypes 用）
valid_params = {
    "n_clusters": 3,
    "max_iter": 10,
    "min_n_moves": 0,
    "n_init": 1,
    "random_state": 42,
    "init": "random",            # 文字列で指定
    "categorical_measure": "hamming",
    "numerical_measure": "euclidean",
    "n_jobs": 1,
    "print_log": False,
    "recompile": False,
    "use_simd": False,
    "gamma": 1.0,
    "max_tol": 0.1
}

# ===================== predict() と compute_score() のテスト =====================
def test_prototypes_predict_normal():
    X = create_valid_X()
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    model.fit(X, categorical)
    pred = model.predict(X, return_distance=False)
    assert isinstance(pred, np.ndarray)
    assert pred.shape[0] == X.shape[0]

def test_prototypes_compute_score_normal():
    X = create_valid_X()
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    model.fit(X, categorical)
    score = model.compute_score(X)
    assert isinstance(score, (np.float32, np.float64, float))
    assert score >= 0

def test_prototypes_predict_without_fit():
    X = create_valid_X()
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.predict(X)
    assert "Model has not been fitted" in str(excinfo.value)

def test_prototypes_predict_X_wrong_columns():
    X = create_valid_X()
    categorical = [0, 1]
    # fit時は4列、予測時に3列を与える
    X_wrong = X[:, :3]
    model = FasterKPrototypes(**valid_params)
    model.fit(X, categorical)
    with pytest.raises(ValueError) as excinfo:
        model.predict(X_wrong)
    assert "same number of columns" in str(excinfo.value)

def test_prototypes_predict_X_not_array():
    X = [[1, 2, 10, 20],
         [2, 3, 11, 21],
         [3, 4, 12, 22],
         [4, 5, 13, 23],
         [5, 1, 14, 24]]
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    model.fit(create_valid_X(), categorical)
    with pytest.raises(ValueError) as excinfo:
        model.predict(X)
    assert "numpy ndarray" in str(excinfo.value)

def test_prototypes_predict_X_not_C_contiguous():
    X = create_valid_X(order='F')
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    model.fit(create_valid_X(), categorical)
    with pytest.raises(ValueError) as excinfo:
        model.predict(X)
    assert "C-order" in str(excinfo.value)
