import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# FasterKPrototypes 用テスト
def test_kprototypes_invalid_n_clusters():
    # n_clusters=1 は条件 (2以上) を満たさないのでエラーになるはず
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=1,  # 無効な値
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
    assert "n_clusters" in str(excinfo.value)

def test_kprototypes_valid_n_clusters():
    # n_clusters=2 は有効な値なのでエラーは出ず、属性にも正しく設定される
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
    assert model.n_clusters == 2

# FasterKModes 用テスト
def test_kmodes_invalid_n_clusters():
    # n_clusters=1 は条件 (2以上) を満たさないのでエラーになるはず
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=1,  # 無効な値
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
    assert "n_clusters" in str(excinfo.value)

def test_kmodes_valid_n_clusters():
    # n_clusters=2 は有効な値なのでエラーは出ず、属性にも正しく設定される
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
    assert model.n_clusters == 2
