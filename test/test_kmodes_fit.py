import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest
import numpy as np
from FasterKModes import FasterKModes

# --- ヘルパー関数 ---
def create_valid_X(n_samples=5, n_features=4, dtype=np.uint8):
    # ユニークな行が n_clusters 以上となるような C-contiguous な配列
    X = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8]
    ], dtype=dtype, order='C')
    return X

def create_valid_init_C(n_clusters, n_features, dtype=np.uint8):
    # X の値域内の値で n_clusters 行 n_features 列の配列
    init_C = np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ], dtype=dtype, order='C')
    # ここでは n_clusters=3 を前提
    return init_C

# 共通のハイパラ（n_clusters=3 とする）
valid_params = {
    "n_clusters": 3,
    "max_iter": 10,
    "min_n_moves": 0,
    "n_init": 1,
    "random_state": 42,
    "init": "random",            # ここでは文字列として指定
    "categorical_measure": "hamming",
    "n_jobs": 1,
    "print_log": False,
    "recompile": False,
    "use_simd": False,
    "max_tol": 0.1
}

# ===================== X に対する検証 =====================

# １．正常なテスト: 正常な X ならエラーなく fit() が実行される
def test_kmodes_fit_normal():
    X = create_valid_X()
    model = FasterKModes(**valid_params)
    # エラーなく fit() が実行され、内部でクラスタが設定されるはず
    model.fit(X)
    assert hasattr(model, "C")

# ２．X は np.array であること（リストの場合エラー）
def test_kmodes_fit_X_not_array():
    X = [[1, 2, 3, 4],
         [2, 3, 4, 5],
         [3, 4, 5, 6],
         [4, 5, 6, 7],
         [5, 6, 7, 8]]
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X)
    assert "must be a numpy array" in str(excinfo.value)

# ３．X は 2 次元配列であること（1次元の場合エラー）
def test_kmodes_fit_X_not_2d():
    X = np.array([1, 2, 3, 4], dtype=np.uint8)
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X)
    assert "2D" in str(excinfo.value)

# ４．X の dtype は uint8 または uint16 であること（異なる型の場合エラー）
def test_kmodes_fit_X_wrong_dtype():
    X = np.array([[1,2,3,4],
                  [2,3,4,5],
                  [3,4,5,6],
                  [4,5,6,7],
                  [5,6,7,8]], dtype=np.float32, order='C')
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X)
    assert "dtype" in str(excinfo.value)

# ５．X は C 配列であること（Fortran オーダーの場合エラー）
def test_kmodes_fit_X_not_C_contiguous():
    X = np.array([[1,2,3,4],
                  [2,3,4,5],
                  [3,4,5,6],
                  [4,5,6,7],
                  [5,6,7,8]], dtype=np.uint8, order='F')
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X)
    assert "order='C'" in str(excinfo.value)

# ６．X のユニークな行数が n_clusters 以上であること（ユニーク行数が足りない場合エラー）
def test_kmodes_fit_not_enough_unique_rows():
    # n_clusters=3 だが、ユニークな行が 2 しかない場合
    X = np.array([
        [1,2,3,4],
        [1,2,3,4],
        [2,3,4,5],
        [2,3,4,5]
    ], dtype=np.uint8, order='C')
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X)
    assert "unique rows" in str(excinfo.value)

# ７．init が設定されている場合（有効な文字列なら正常に実行）
def test_kmodes_fit_with_valid_init():
    X = create_valid_X()
    params = valid_params.copy()
    params["init"] = "k-means++"  # 有効な文字列
    model = FasterKModes(**params)
    model.fit(X)
    assert hasattr(model, "C")


# ===================== init_C が設定されている場合の検証 =====================

# ８．init_C は np.array であること（リストの場合エラー）
def test_kmodes_fit_init_C_not_array():
    X = create_valid_X()
    init_C = [[1,2,3,4],
              [2,3,4,5],
              [3,4,5,6]]  # リスト
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, init_C=init_C)
    assert "init_C must be a numpy array" in str(excinfo.value)

# ９．init_C は 2 次元配列であること（1次元の場合エラー）
def test_kmodes_fit_init_C_not_2d():
    X = create_valid_X()
    init_C = np.array([1,2,3,4], dtype=np.uint8, order='C')  # 1D
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, init_C=init_C)
    assert "2D" in str(excinfo.value)

# １０．init_C の列数が X と同じであること（列数が異なる場合エラー）
def test_kmodes_fit_init_C_wrong_columns():
    X = create_valid_X(n_features=4)
    # X は 4 列であるが、init_C は 3 列にする
    init_C = np.array([[1,2,3],
                       [2,3,4],
                       [3,4,5]], dtype=np.uint8, order='C')
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, init_C=init_C)
    assert "same number of columns" in str(excinfo.value)

# １１．init_C の型が X と同じであること（dtype が異なる場合エラー）
def test_kmodes_fit_init_C_wrong_dtype():
    X = create_valid_X(dtype=np.uint8)
    # init_C の型を np.uint16 にする（異なる型）
    init_C = np.array([[1,2,3,4],
                       [2,3,4,5],
                       [3,4,5,6]], dtype=np.uint16, order='C')
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, init_C=init_C)
    assert "must match X's dtype" in str(excinfo.value)

# １２．init_C の最大値が X の最大値を超えていないこと（超えている場合エラー）
def test_kmodes_fit_init_C_exceeds_X_max():
    X = create_valid_X()  # ここでは X の最大値は 8
    # init_C に 100 など、X の最大値 (8) を超える値を含める
    init_C = np.array([[1,2,3,4],
                       [2,3,4,5],
                       [100,4,5,6]], dtype=np.uint8, order='C')
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, init_C=init_C)
    assert "contains values outside the range" in str(excinfo.value)

# １３．init_C の行数が n_clusters と一致すること（行数が異なる場合エラー）
def test_kmodes_fit_init_C_wrong_rows():
    X = create_valid_X()
    # n_clusters=3 であるのに、init_C の行数を 2 にする
    init_C = np.array([[1,2,3,4],
                       [2,3,4,5]], dtype=np.uint8, order='C')
    model = FasterKModes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, init_C=init_C)
    assert "same number of rows" in str(excinfo.value)

# １４．init が Callable ではないこと（init_C が設定されている場合、init は文字列でなければならない）
def test_kmodes_fit_init_C_with_callable_init():
    X = create_valid_X()
    init_C = create_valid_init_C(n_clusters=valid_params["n_clusters"], n_features=X.shape[1])
    params = valid_params.copy()
    # init を callable に設定（init_C がある場合はエラーとなることを期待）
    params["init"] = lambda X, n_clusters: (X[:n_clusters], X[:n_clusters])
    model = FasterKModes(**params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, init_C=init_C)
    assert "Cannot provide both a custom init function" in str(excinfo.value)

# 正常系：init_C が設定されている場合、すべての条件を満たすと fit() が実行される
def test_kmodes_fit_valid_init_C():
    X = create_valid_X()
    init_C = create_valid_init_C(n_clusters=valid_params["n_clusters"], n_features=X.shape[1])
    params = valid_params.copy()
    # init は文字列で設定（例："random"）
    params["init"] = "random"
    model = FasterKModes(**params)
    model.fit(X, init_C=init_C)
    # fit() 後、C が設定され、形状が正しいことを確認
    assert hasattr(model, "C")
    assert model.C.shape[0] == params["n_clusters"]
    assert model.C.shape[1] == X.shape[1]
