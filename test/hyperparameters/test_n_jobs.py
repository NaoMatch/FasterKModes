import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# FasterKModes 用テスト

def test_kmodes_valid_n_jobs():
    """n_jobs に正の整数を指定した場合に正常動作するかをテストする。"""
    model = FasterKModes(
            n_jobs=10, # 正常な型と範囲
         )
    assert model.n_jobs == 10

def test_kmodes_invalid_range_n_jobs():
    """n_jobs に 0 以下の値を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_jobs=0, # 型はあっているが、値の範囲が無効
        )
    assert "n_jobs" in str(excinfo.value)

def test_kmodes_invalid_type_n_jobs():
    """n_jobs に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_jobs="10",  # 型があっていない
        )
    assert "n_jobs" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_n_jobs():
    """n_jobs に正の整数を指定した場合に正常動作するかをテストする。"""
    model = FasterKPrototypes(
            n_jobs=10, # 正常な型と範囲
         )
    assert model.n_jobs == 10

def test_kprototypes_invalid_range_n_jobs():
    """n_jobs に 0 以下の値を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_jobs=0, # 型はあっているが、値の範囲が無効
        )
    assert "n_jobs" in str(excinfo.value)

def test_kprototypes_invalid_type_n_jobs():
    """n_jobs に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_jobs="10",  # 型があっていない
        )
    assert "n_jobs" in str(excinfo.value)
