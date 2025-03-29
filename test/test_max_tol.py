import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# ======= FasterKPrototypes の max_tol テスト =======

def test_kprototypes_invalid_max_tol_negative():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
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
            max_tol=-0.1  # 負の値は無効
        )
    assert "max_tol" in str(excinfo.value)

def test_kprototypes_invalid_max_tol_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
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
            max_tol="low"  # 型が不正
        )
    assert "max_tol" in str(excinfo.value)

def test_kprototypes_valid_max_tol_zero():
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
        max_tol=0  # 有効な値
    )
    assert model.max_tol == 0

def test_kprototypes_valid_max_tol_positive():
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
        max_tol=1.5  # 有効な値
    )
    assert model.max_tol == 1.5

def test_kprototypes_valid_max_tol_none():
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
        max_tol=None  # None は有効
    )
    assert model.max_tol is None

# ======= FasterKModes の max_tol テスト =======

def test_kmodes_invalid_max_tol_negative():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
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
            max_tol=-1.0  # 負の値は無効
        )
    assert "max_tol" in str(excinfo.value)

def test_kmodes_invalid_max_tol_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
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
            max_tol="medium"  # 型が不正
        )
    assert "max_tol" in str(excinfo.value)

def test_kmodes_valid_max_tol_zero():
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
        max_tol=0  # 有効な値
    )
    assert model.max_tol == 0

def test_kmodes_valid_max_tol_positive():
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
        max_tol=2.5  # 有効な値
    )
    assert model.max_tol == 2.5

def test_kmodes_valid_max_tol_none():
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
        max_tol=None  # None は有効
    )
    assert model.max_tol is None
