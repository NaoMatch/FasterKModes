import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# ======= FasterKPrototypes の random_state テスト =======

def test_kprototypes_invalid_random_state_string():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state="invalid",  # 文字列は無効
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
    assert "random_state must be either None or an integer" in str(excinfo.value)

def test_kprototypes_invalid_random_state_list():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=[1, 2, 3],  # リストは無効
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
    assert "random_state must be either None or an integer" in str(excinfo.value)

def test_kprototypes_valid_random_state_int():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,  # 有効な整数
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
    assert model.random_state == 42

def test_kprototypes_valid_random_state_none():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=None,  # None は有効
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
    assert model.random_state is None

# ======= FasterKModes の random_state テスト =======

def test_kmodes_invalid_random_state_string():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state="invalid",  # 文字列は無効
            init="random",
            categorical_measure="hamming",
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "random_state must be either None or an integer" in str(excinfo.value)

def test_kmodes_invalid_random_state_list():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=[1, 2, 3],  # リストは無効
            init="random",
            categorical_measure="hamming",
            n_jobs=1,
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "random_state must be either None or an integer" in str(excinfo.value)

def test_kmodes_valid_random_state_int():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,  # 有効な整数
        init="random",
        categorical_measure="hamming",
        n_jobs=1,
        print_log=False,
        recompile=False,
        use_simd=False,
        max_tol=0.1
    )
    assert model.random_state == 42

def test_kmodes_valid_random_state_none():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=None,  # None は有効
        init="random",
        categorical_measure="hamming",
        n_jobs=1,
        print_log=False,
        recompile=False,
        use_simd=False,
        max_tol=0.1
    )
    assert model.random_state is None
