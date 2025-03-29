import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# --- valid callable for FasterKPrototypes: 引数名は ["Xcat", "Ccat"] ---
def valid_categorical_measure_prototypes(Xcat, Ccat):
    # ダミー実装（チェック対象外）
    return 0

# --- invalid callable for FasterKPrototypes: 引数が1つ ---
def invalid_categorical_measure_prototypes_one_arg(Xcat):
    return 0

# --- invalid callable for FasterKPrototypes: 引数名が異なる ---
def invalid_categorical_measure_prototypes_wrong_names(X, Ccat):
    return 0

# --- valid callable for FasterKModes: 引数名は ["Xcat", "Ccat"] ---
def valid_categorical_measure_kmodes(Xcat, Ccat):
    return 0

# --- invalid callable for FasterKModes: 引数が1つ ---
def invalid_categorical_measure_kmodes_one_arg(X):
    return 0

# --- invalid callable for FasterKModes: 引数名が異なる ---
def invalid_categorical_measure_kmodes_wrong_names(X, C):
    return 0

# ===== FasterKPrototypes のテスト =====

def test_kprototypes_invalid_categorical_measure_string():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure="not_hamming",  # 無効な文字列
            numerical_measure="euclidean",
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "categorical_measure" in str(excinfo.value)

def test_kprototypes_invalid_categorical_measure_callable_one_arg():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure=invalid_categorical_measure_prototypes_one_arg,  # 引数が1つなので無効
            numerical_measure="euclidean",
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "Custom categorical_measure function" in str(excinfo.value)

def test_kprototypes_invalid_categorical_measure_callable_wrong_names():
    with pytest.raises(ValueError) as excinfo:
        # この関数は引数名が ["X", "Ccat"] となっており、要求される ["Xcat", "Ccat"] とは異なるため無効
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure=invalid_categorical_measure_prototypes_wrong_names,
            numerical_measure="euclidean",
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "Custom categorical_measure function" in str(excinfo.value)

def test_kprototypes_invalid_categorical_measure_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure=123,  # 型が不正
            numerical_measure="euclidean",
            n_jobs=1,
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "categorical_measure" in str(excinfo.value)

def test_kprototypes_valid_categorical_measure_string():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",  # 有効な文字列
        numerical_measure="euclidean",
        n_jobs=1,
        print_log=False,
        gamma=1.0,
        recompile=False,
        use_simd=True,
        max_tol=0.1
    )
    assert model.categorical_measure == "hamming"

def test_kprototypes_valid_categorical_measure_callable():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure=valid_categorical_measure_prototypes,  # 正しいシグネチャの callable
        numerical_measure="euclidean",
        n_jobs=1,
        print_log=False,
        gamma=1.0,
        recompile=False,
        use_simd=True,
        max_tol=0.1
    )
    assert callable(model.categorical_measure)

# ===== FasterKModes のテスト =====

def test_kmodes_invalid_categorical_measure_string():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure="not_hamming",  # 無効な文字列
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "categorical_measure" in str(excinfo.value)

def test_kmodes_invalid_categorical_measure_callable_one_arg():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure=invalid_categorical_measure_kmodes_one_arg,  # 引数が1つなので無効
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "Custom categorical_measure function" in str(excinfo.value)

def test_kmodes_invalid_categorical_measure_callable_wrong_names():
    with pytest.raises(ValueError) as excinfo:
        # この関数は引数名が ["Xcat", "Ccat"] となっており、要求される ["X", "C"] とは異なるため無効
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure=invalid_categorical_measure_kmodes_wrong_names,
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "Custom categorical_measure function" in str(excinfo.value)

def test_kmodes_invalid_categorical_measure_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure=123,  # 型が不正
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "categorical_measure" in str(excinfo.value)

def test_kmodes_valid_categorical_measure_string():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",  # 有効な文字列
        n_jobs=1,
        print_log=False,
        recompile=False,
        use_simd=False,
        max_tol=0.1
    )
    assert model.categorical_measure == "hamming"

def test_kmodes_valid_categorical_measure_callable():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure=valid_categorical_measure_kmodes,  # 正しいシグネチャの callable
        n_jobs=1,
        print_log=False,
        recompile=False,
        use_simd=False,
        max_tol=0.1
    )
    assert callable(model.categorical_measure)
