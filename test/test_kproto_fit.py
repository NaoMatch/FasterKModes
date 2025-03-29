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

# ===================== fit() のテスト =====================

def test_prototypes_fit_normal():
    X = create_valid_X()
    # ここでは、最初の2列をカテゴリカル、残りを数値特徴とする
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    model.fit(X, categorical)
    # fit後、クラスタ情報として Ccat, Cnum が設定されることを確認
    assert hasattr(model, "Ccat")
    assert hasattr(model, "Cnum")

def test_prototypes_fit_X_not_array():
    X = [[1, 2, 10, 20],
         [2, 3, 11, 21],
         [3, 4, 12, 22],
         [4, 5, 13, 23],
         [5, 1, 14, 24]]
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical)
    assert "X must be a numpy ndarray" in str(excinfo.value)

def test_prototypes_fit_X_not_2d():
    X = np.array([1, 2, 10, 20], dtype=np.uint8)
    categorical = [0]  # ダミーとして指定
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical)
    assert "2-dimensional" in str(excinfo.value)

def test_prototypes_fit_X_wrong_dtype():
    # 許容外の dtype（例：np.int32）を指定
    X = np.array([
        [1, 2, 10, 20],
        [2, 3, 11, 21],
        [3, 4, 12, 22],
        [4, 5, 13, 23],
        [5, 1, 14, 24]
    ], dtype=np.int32, order='C')
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical)
    assert "dtype" in str(excinfo.value)

def test_prototypes_fit_X_not_C_contiguous():
    X = np.array([
        [1, 2, 10, 20],
        [2, 3, 11, 21],
        [3, 4, 12, 22],
        [4, 5, 13, 23],
        [5, 1, 14, 24]
    ], dtype=np.uint8, order='F')
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical)
    assert "order='C'" in str(excinfo.value)

def test_prototypes_fit_not_enough_unique_rows():
    # n_clusters=3 に対し、ユニークな行数が 2 の場合
    X = np.array([
        [1, 2, 10, 20],
        [1, 2, 10, 20],
        [2, 3, 11, 21]
    ], dtype=np.uint8, order='C')
    categorical = [0, 1]
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical)
    assert "unique rows" in str(excinfo.value)

def test_prototypes_fit_with_valid_init():
    X = create_valid_X()
    categorical = [0, 1]
    params = valid_params.copy()
    params["init"] = "k-means++"
    model = FasterKPrototypes(**params)
    model.fit(X, categorical)
    assert hasattr(model, "Ccat")
    assert hasattr(model, "Cnum")

# -------------------- init_Ccat, init_Cnum のテスト --------------------
def test_prototypes_fit_init_Ccat_not_array():
    X = create_valid_X()
    categorical = [0, 1]
    init_Ccat = [[1, 2],
                 [2, 3],
                 [3, 4]]  # リスト形式
    init_Cnum = create_valid_init_Cnum()
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Ccat must be a numpy ndarray" in str(excinfo.value)

def test_prototypes_fit_init_Ccat_not_2d():
    X = create_valid_X()
    categorical = [0, 1]
    init_Ccat = np.array([1, 2], dtype=np.uint8, order='C')  # 1次元
    init_Cnum = create_valid_init_Cnum()
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "2-dimensional" in str(excinfo.value)

def test_prototypes_fit_init_Ccat_wrong_columns():
    X = create_valid_X()
    categorical = [0, 1]
    # カテゴリカル部分は2列必要だが、1列のみを与える
    init_Ccat = np.array([[1],
                          [2],
                          [3]], dtype=np.uint8, order='C')
    init_Cnum = create_valid_init_Cnum()
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "same number of columns" in str(excinfo.value)

def test_prototypes_fit_init_Ccat_wrong_dtype():
    X = create_valid_X(dtype=np.uint8)
    categorical = [0, 1]
    # 型を np.uint16 にして不整合を発生させる
    init_Ccat = np.array([
        [1, 2],
        [2, 3],
        [3, 4]
    ], dtype=np.uint16, order='C')
    init_Cnum = create_valid_init_Cnum()
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Ccat must have the same dtype" in str(excinfo.value)

def test_prototypes_fit_init_Ccat_exceeds_X_max():
    X = create_valid_X()  # X のカテゴリカル部分の最大値は各列で (例: 1～5)
    categorical = [0, 1]
    # X の値域を超える値（例：100）を含む
    init_Ccat = np.array([
        [1, 2],
        [2, 3],
        [100, 4]
    ], dtype=np.uint8, order='C')
    init_Cnum = create_valid_init_Cnum()
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "invalid values for categorical features" in str(excinfo.value)

def test_prototypes_fit_init_Ccat_wrong_rows():
    X = create_valid_X()
    categorical = [0, 1]
    # n_clusters=3 に対し、2 行のみ与える
    init_Ccat = np.array([
        [1, 2],
        [2, 3]
    ], dtype=np.uint8, order='C')
    init_Cnum = create_valid_init_Cnum()
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "same number of rows" in str(excinfo.value)

def test_prototypes_fit_init_Ccat_without_Cnum():
    X = create_valid_X()
    categorical = [0, 1]
    init_Ccat = create_valid_init_Ccat()
    # init_Cnum を与えずに fit を実行（エラーが発生することを期待）
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat)
    assert "init_Ccat and init_Cnum must either both be None or both be provided" in str(excinfo.value)

def test_prototypes_fit_valid_init_C():
    X = create_valid_X()
    categorical = [0, 1]
    init_Ccat = create_valid_init_Ccat()
    init_Cnum = create_valid_init_Cnum()
    model = FasterKPrototypes(**valid_params)
    model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    # fit後、Ccat, Cnum が設定され、形状が正しいことを確認
    assert hasattr(model, "Ccat")
    assert hasattr(model, "Cnum")
    assert model.Ccat.shape[0] == valid_params["n_clusters"]
    assert model.Ccat.shape[1] == len(categorical)
    num_features = X.shape[1] - len(categorical)
    assert model.Cnum.shape[0] == valid_params["n_clusters"]
    assert model.Cnum.shape[1] == num_features


# -------------------- init_Cnum のテスト --------------------
def test_prototypes_fit_init_Cnum_not_array():
    X = create_valid_X()
    categorical = [0, 1]
    init_Ccat = create_valid_init_Ccat()  # 正常なカテゴリカル初期値
    init_Cnum = [[10.0, 20.0],
                 [11.0, 21.0],
                 [12.0, 22.0]]  # リスト形式（NumPy 配列でない）
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Cnum must be a numpy ndarray" in str(excinfo.value)

def test_prototypes_fit_init_Cnum_not_2d():
    X = create_valid_X()
    categorical = [0, 1]
    init_Ccat = create_valid_init_Ccat()
    init_Cnum = np.array([10.0, 20.0], dtype=np.float32, order='C')  # 1次元配列
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "2-dimensional" in str(excinfo.value)

def test_prototypes_fit_init_Cnum_wrong_columns():
    X = create_valid_X()
    categorical = [0, 1]
    init_Ccat = create_valid_init_Ccat()
    # 数値特徴は X の列数 - カテゴリカル数 = 2 列であるはずだが、1列のみの配列を与える
    init_Cnum = np.array([[10.0],
                          [11.0],
                          [12.0]], dtype=np.float32, order='C')
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "same number of columns" in str(excinfo.value)

def test_prototypes_fit_init_Cnum_wrong_dtype():
    X = create_valid_X(dtype=np.uint8)
    categorical = [0, 1]
    init_Ccat = create_valid_init_Ccat()
    # 数値初期値は float 型である必要があるが、int 型の配列を与える
    init_Cnum = np.array([[10, 20],
                          [11, 21],
                          [12, 22]], dtype=np.int32, order='C')
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "init_Cnum must contain float values" in str(excinfo.value)

def test_prototypes_fit_init_Cnum_wrong_rows():
    X = create_valid_X()
    categorical = [0, 1]
    init_Ccat = create_valid_init_Ccat()
    # n_clusters=3 に対し、init_Cnum の行数を 2 行にする（不整合）
    init_Cnum = np.array([[10.0, 20.0],
                          [11.0, 21.0]], dtype=np.float32, order='C')
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Ccat=init_Ccat, init_Cnum=init_Cnum)
    assert "same number of rows" in str(excinfo.value)

def test_prototypes_fit_init_Cnum_without_Ccat():
    X = create_valid_X()
    categorical = [0, 1]
    init_Cnum = create_valid_init_Cnum()
    # init_Ccat を与えず、init_Cnum のみを指定した場合（両方とも None でなければならない）
    model = FasterKPrototypes(**valid_params)
    with pytest.raises(ValueError) as excinfo:
        model.fit(X, categorical, init_Cnum=init_Cnum)
    assert "init_Ccat and init_Cnum must either both be None or both be provided" in str(excinfo.value)
    
