import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import os
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# ======= FasterKPrototypes の n_jobs テスト =======

def test_kprototypes_invalid_n_jobs_zero():
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
            n_jobs=0,  # 0 は無効
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "n_jobs" in str(excinfo.value)

def test_kprototypes_invalid_n_jobs_negative():
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
            n_jobs=-3,  # 負の値は無効
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "n_jobs" in str(excinfo.value)

def test_kprototypes_invalid_n_jobs_wrong_type():
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
            n_jobs="2",  # 文字列は無効
            print_log=False,
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "n_jobs" in str(excinfo.value)

def test_kprototypes_valid_n_jobs_integer():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",
        numerical_measure="euclidean",
        n_jobs=2,  # 有効な整数
        print_log=False,
        gamma=1.0,
        recompile=False,
        use_simd=True,
        max_tol=0.1
    )
    expected = 2 if 2 <= os.cpu_count() else os.cpu_count()
    assert model.n_jobs == expected

def test_kprototypes_valid_n_jobs_none():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",
        numerical_measure="euclidean",
        n_jobs=None,  # None は有効
        print_log=False,
        gamma=1.0,
        recompile=False,
        use_simd=True,
        max_tol=0.1
    )
    assert model.n_jobs == os.cpu_count()

# ======= FasterKModes の n_jobs テスト =======

def test_kmodes_invalid_n_jobs_zero():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure="hamming",
            n_jobs=0,  # 0 は無効
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "n_jobs" in str(excinfo.value)

def test_kmodes_invalid_n_jobs_negative():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure="hamming",
            n_jobs=-1,  # 負の値は無効
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "n_jobs" in str(excinfo.value)

def test_kmodes_invalid_n_jobs_wrong_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=10,
            min_n_moves=0,
            n_init=1,
            random_state=42,
            init="random",
            categorical_measure="hamming",
            n_jobs="one",  # 文字列は無効
            print_log=False,
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "n_jobs" in str(excinfo.value)

def test_kmodes_valid_n_jobs_integer():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",
        n_jobs=2,  # 有効な整数
        print_log=False,
        recompile=False,
        use_simd=False,
        max_tol=0.1
    )
    expected = 2 if 2 <= os.cpu_count() else os.cpu_count()
    assert model.n_jobs == expected

def test_kmodes_valid_n_jobs_none():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",
        n_jobs=None,  # None は有効
        print_log=False,
        recompile=False,
        use_simd=False,
        max_tol=0.1
    )
    assert model.n_jobs == os.cpu_count()
