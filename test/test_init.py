import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# 正しいシグネチャの init 関数
def valid_init(X, n_clusters):
    # ダミーの初期化処理（返り値はチェック対象外）
    return X[:n_clusters], X[:n_clusters]

# 間違ったシグネチャ（引数が1つ）の init 関数
def invalid_init_one_arg(X):
    return X

# 間違ったシグネチャ（引数名が異なる）の init 関数
def invalid_init_wrong_names(a, b):
    return a, b

# ===== FasterKPrototypes のテスト =====

def test_kprototypes_invalid_init_string():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="invalid",  # 無効な文字列
            categorical_measure="hamming",
            numerical_measure="euclidean",
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "init must be one of" in str(excinfo.value)

def test_kprototypes_invalid_init_callable_one_arg():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init=invalid_init_one_arg,  # 引数が1つなので無効
            categorical_measure="hamming",
            numerical_measure="euclidean",
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "Custom init function must accept exactly two arguments" in str(excinfo.value)

def test_kprototypes_invalid_init_callable_wrong_names():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init=invalid_init_wrong_names,  # 引数名が "a", "b" なので無効
            categorical_measure="hamming",
            numerical_measure="euclidean",
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "Custom init function must accept exactly two arguments" in str(excinfo.value)

def test_kprototypes_invalid_init_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init=123,  # 数値は無効
            categorical_measure="hamming",
            numerical_measure="euclidean",
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "init must be a string or a callable function" in str(excinfo.value)

def test_kprototypes_valid_init_string():
    # "random" は有効な文字列
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",
        numerical_measure="euclidean",
        n_jobs=1,
        print_log=False,
        gamma=1.0,
        recompile=False,
        use_simd=True,
        max_tol=0.1
    )
    assert model.init == "random"

def test_kprototypes_valid_init_callable():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init=valid_init,  # 正しいシグネチャの callable
        categorical_measure="hamming",
        numerical_measure="euclidean",
        n_jobs=1,
        print_log=False,
        gamma=1.0,
        recompile=False,
        use_simd=True,
        max_tol=0.1
    )
    # テストではシグネチャチェックのみのため、model.init は valid_init そのものとなる
    assert callable(model.init)

# ===== FasterKModes のテスト =====

def test_kmodes_invalid_init_string():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="invalid",  # 無効な文字列
            categorical_measure="hamming",
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "init must be one of" in str(excinfo.value)

def test_kmodes_invalid_init_callable_one_arg():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init=invalid_init_one_arg,  # 引数が1つなので無効
            categorical_measure="hamming",
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "Custom init function must accept exactly two arguments" in str(excinfo.value)

def test_kmodes_invalid_init_callable_wrong_names():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init=invalid_init_wrong_names,  # 引数名が "a", "b" なので無効
            categorical_measure="hamming",
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "Custom init function must accept exactly two arguments" in str(excinfo.value)

def test_kmodes_invalid_init_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init=123,  # 数値は無効
            categorical_measure="hamming",
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "init must be a string or a callable function" in str(excinfo.value)

def test_kmodes_valid_init_string():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",
        n_jobs=1,
        print_log=False,
        recompile=False,
        use_simd=False,
        max_tol=0.1
    )
    assert model.init == "random"

def test_kmodes_valid_init_callable():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init=valid_init,  # 正しいシグネチャの callable
        categorical_measure="hamming",
        n_jobs=1,
        print_log=False,
        recompile=False,
        use_simd=False,
        max_tol=0.1
    )
    assert callable(model.init)
