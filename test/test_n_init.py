import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# ======= FasterKPrototypes の n_init テスト =======

def test_kprototypes_invalid_n_init_zero():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=0,  # 無効な値 (1以上でなければならない)
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
    assert "n_init" in str(excinfo.value)

def test_kprototypes_invalid_n_init_negative():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=-5,  # 負の値は無効
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
    assert "n_init" in str(excinfo.value)

def test_kprototypes_invalid_n_init_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init="one",  # 型が不正
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
    assert "n_init" in str(excinfo.value)

def test_kprototypes_valid_n_init():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=5,  # 有効な値
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
    assert model.n_init == 5

# ======= FasterKModes の n_init テスト =======

def test_kmodes_invalid_n_init_zero():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=0,  # 無効な値
            random_state=42,
            init="random",
            categorical_measure="hamming",
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "n_init" in str(excinfo.value)

def test_kmodes_invalid_n_init_negative():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=-3,  # 負の値は無効
            random_state=42,
            init="random",
            categorical_measure="hamming",
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "n_init" in str(excinfo.value)

def test_kmodes_invalid_n_init_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init="ten",  # 型が不正
            random_state=42,
            init="random",
            categorical_measure="hamming",
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "n_init" in str(excinfo.value)

def test_kmodes_valid_n_init():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=3,  # 有効な値
        random_state=42,
        init="random",
        categorical_measure="hamming",
        n_jobs=1,
        print_log=False,
        recompile=False,
        use_simd=False,
        max_tol=0.1
    )
    assert model.n_init == 3
