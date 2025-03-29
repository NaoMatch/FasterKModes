import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from FasterKModes import FasterKModes

# ヘルパー関数：有効な入力データ X（uint8、2次元、C 配列）を作成
def create_valid_X(n_samples=5, n_features=4, dtype=np.uint8, order="C"):
    # ユニークな行が n_clusters(例:3) 以上になるように作成
    X = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8]
    ], dtype=dtype, order=order)
    return X

# 共通のハイパラ（n_clusters=3 とする）
valid_params = {
    "n_clusters": 3,
    "max_iter": 10,
    "min_n_moves": 0,
    "n_init": 1,
    "random_state": 42,
    "init": "random",  # ここでは文字列で指定
    "categorical_measure": "hamming",
    "n_jobs": 1,
    "print_log": False,
    "recompile": False,
    "use_simd": False,
    "max_tol": 0.1
}

# -------------------- 正常系テスト --------------------

def test_kmodes_predict_normal():
    # 正常な入力 X で fit() → predict() が正常に実行される
    X = create_valid_X()
    model = FasterKModes(**valid_params)
    model.fit(X)
    # predict() の返り値は np.array であり、行数は X と同じ
    pred = model.predict(X)
    assert isinstance(pred, np.ndarray)
    assert pred.shape[0] == X.shape[0]

def test_kmodes_compute_score_normal():
    # 正常な入力 X で fit() → compute_score() が正常に実行される
    X = create_valid_X()
    model = FasterKModes(**valid_params)
    model.fit(X)
    # predict() の返り値は np.array であり、行数は X と同じ
    score = model.compute_score(X)
    assert isinstance(score, np.int32) | isinstance(score, np.int64)
    assert score >= 0

# -------------------- 異常系テスト --------------------

def test_kmodes_predict_without_fit():
    # fit() を呼ばずに predict() を実行するとエラーとなる
    X = create_valid_X()
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.predict(X)
    assert "Model has not been fitted" in str(excinfo.value)

def test_kmodes_predict_X_zero_rows():
    # X が 0 行の場合、エラーとなる（行数チェック）
    # ※ __validate_predict_X 内で明示的なチェックはないが、ここでは入力が空の場合のエラーを期待
    X = np.empty((0, 4), dtype=np.uint8, order='C')
    model = FasterKModes(**valid_params)
    model.fit(create_valid_X())  # fit() には正常なデータを使用
    with pytest.raises(ValueError) as excinfo:
        model.predict(X)
    # 例: "must have the same number of columns" とはならないので、任意のエラーメッセージをチェック
    assert "X must have" in str(excinfo.value) or "negative" in str(excinfo.value)

def test_kmodes_predict_X_wrong_columns():
    # X の列数が fit 時と異なる場合、エラーとなる
    X = create_valid_X(n_samples=5, n_features=4)
    # fit 時は 4 列で学習しているので、ここでは 3 列の X を用意
    X_wrong = X[:, :3]
    model = FasterKModes(**valid_params)
    model.fit(X)
    with pytest.raises(ValueError) as excinfo:
        model.predict(X_wrong)
    assert "same number of columns" in str(excinfo.value)

def test_kmodes_predict_X_not_array():
    # X が np.array でない場合、エラーとなる
    X = [[1,2,3,4],
         [2,3,4,5],
         [3,4,5,6],
         [4,5,6,7],
         [5,6,7,8]]
    model = FasterKModes(**valid_params)
    model.fit(create_valid_X())
    with pytest.raises(ValueError) as excinfo:
        model.predict(X)
    assert "must be a numpy array" in str(excinfo.value)

def test_kmodes_predict_X_wrong_dtype():
    # X の dtype が fit 時と異なる場合、エラーとなる
    X = create_valid_X(dtype=np.uint8)
    # fit 時は uint8 で学習しているので、ここでは dtype を uint16 に変更
    X_wrong = X.astype(np.uint16)
    model = FasterKModes(**valid_params)
    model.fit(X)
    with pytest.raises(ValueError) as excinfo:
        model.predict(X_wrong)
    assert "does not match the model's trained input dtype" in str(excinfo.value)

def test_kmodes_predict_X_not_C_contiguous():
    # X が C 配列でない場合（Fortran オーダーの場合）エラーとなる
    X = create_valid_X(order='F')
    model = FasterKModes(**valid_params)
    model.fit(create_valid_X())
    with pytest.raises(ValueError) as excinfo:
        model.predict(X)
    assert "X must be C-order" in str(excinfo.value)
