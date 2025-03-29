import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from FasterKPrototypes import FasterKPrototypes

# --- valid callable: 引数名は ["Xnum", "Cnum"] ---
def valid_numerical_measure(Xnum, Cnum):
    # ダミー実装（返り値はチェック対象外）
    return 0

# --- invalid callable: 引数が1つ ---
def invalid_numerical_measure_one_arg(X):
    return 0

# --- invalid callable: 引数名が異なる（例: ["X", "C"]） ---
def invalid_numerical_measure_wrong_names(X, C):
    return 0

def test_kprototypes_invalid_numerical_measure_string():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure="hamming",
            numerical_measure="manhattan",  # 無効な文字列
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "numerical_measure must be one of" in str(excinfo.value)

def test_kprototypes_invalid_numerical_measure_callable_one_arg():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure="hamming",
            numerical_measure=invalid_numerical_measure_one_arg,  # 引数が1つなので無効
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "Custom numerical_measure function must accept exactly two arguments" in str(excinfo.value)

def test_kprototypes_invalid_numerical_measure_callable_wrong_names():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure="hamming",
            numerical_measure=invalid_numerical_measure_wrong_names,  # 引数名が ["Xnum", "Cnum"] なので無効
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "Custom numerical_measure function must accept exactly two arguments" in str(excinfo.value)

def test_kprototypes_invalid_numerical_measure_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure="hamming",
            numerical_measure=123,  # 数値は無効
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "numerical_measure must be a string" in str(excinfo.value)

def test_kprototypes_valid_numerical_measure_string():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",
        numerical_measure="euclidean",  # 有効な文字列
        n_jobs=1,
        print_log=False,
        gamma=1.0,
        recompile=False,
        use_simd=True,
        max_tol=0.1
    )
    assert model.numerical_measure == "euclidean"

def test_kprototypes_valid_numerical_measure_callable():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",
        numerical_measure=valid_numerical_measure,  # 有効な callable
        n_jobs=1,
        print_log=False,
        gamma=1.0,
        recompile=False,
        use_simd=True,
        max_tol=0.1
    )
    assert callable(model.numerical_measure)
