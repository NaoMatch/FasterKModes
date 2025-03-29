import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# ======= FasterKPrototypes の max_iter テスト =======

def test_kprototypes_invalid_max_iter_zero():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter=0,  # 0 は無効（1以上が必要）
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
    assert "max_iter" in str(excinfo.value)

def test_kprototypes_invalid_max_iter_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=2,
            max_iter="ten",  # 型が不正
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
    assert "max_iter" in str(excinfo.value)

def test_kprototypes_valid_max_iter():
    model = FasterKPrototypes(
        n_clusters=2,
        max_iter=10,  # 有効な値
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
    assert model.max_iter == 10

# ======= FasterKModes の max_iter テスト =======

def test_kmodes_invalid_max_iter_zero():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter=0,  # 0 は無効（1以上が必要）
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
    assert "max_iter" in str(excinfo.value)

def test_kmodes_invalid_max_iter_type():
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=2,
            max_iter="ten",  # 型が不正
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
    assert "max_iter" in str(excinfo.value)

def test_kmodes_valid_max_iter():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,  # 有効な値
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
    assert model.max_iter == 10
