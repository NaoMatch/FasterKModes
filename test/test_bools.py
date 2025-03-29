import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# ======= FasterKPrototypes のテスト =======

def test_kprototypes_invalid_print_log():
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
            print_log="yes",  # Boolean でないので無効
            gamma=1.0,
            recompile=False,
            use_simd=True,
            max_tol=0.1
        )
    assert "print_log" in str(excinfo.value)

def test_kprototypes_invalid_recompile():
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
            recompile="no",  # Boolean でないので無効
            use_simd=True,
            max_tol=0.1
        )
    assert "recompile" in str(excinfo.value)

def test_kprototypes_invalid_use_simd():
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
            use_simd=123,  # Boolean でないので無効
            max_tol=0.1
        )
    assert "use_simd" in str(excinfo.value)

def test_kprototypes_valid_booleans():
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
        print_log=True,
        gamma=1.0,
        recompile=True,
        use_simd=True,
        max_tol=0.1
    )
    assert isinstance(model.print_log, bool)
    assert isinstance(model.recompile, bool)
    assert isinstance(model.use_simd, bool)

# ======= FasterKModes のテスト =======

def test_kmodes_invalid_print_log():
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
            print_log="true",  # Boolean でないので無効
            recompile=False,
            use_simd=False,
            max_tol=0.1
        )
    assert "print_log" in str(excinfo.value)

def test_kmodes_invalid_recompile():
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
            recompile=1,  # Boolean でないので無効
            use_simd=False,
            max_tol=0.1
        )
    assert "recompile" in str(excinfo.value)

def test_kmodes_invalid_use_simd():
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
            use_simd="False",  # Boolean でないので無効
            max_tol=0.1
        )
    assert "use_simd" in str(excinfo.value)

def test_kmodes_valid_booleans():
    model = FasterKModes(
        n_clusters=2,
        max_iter=10,
        min_n_moves=0,
        n_init=1,
        random_state=42,
        init="random",
        categorical_measure="hamming",
        n_jobs=1,
        print_log=True,
        recompile=True,
        use_simd=False,
        max_tol=0.1
    )
    assert isinstance(model.print_log, bool)
    assert isinstance(model.recompile, bool)
    assert isinstance(model.use_simd, bool)
