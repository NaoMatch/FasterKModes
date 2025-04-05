import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# FasterKModes 用テスト

def test_kmodes_valid_max_tol():
    """max_tol に整数または浮動小数点の正の値を指定した場合に正常動作するかをテストする。"""
    model = FasterKModes(
            max_tol=10, # 正常な型と範囲
         )
    assert model.max_tol == 10

    model = FasterKModes(
            max_tol=10.0, # 正常な型と範囲
         )
    assert model.max_tol == 10.0

def test_kmodes_invalid_range_max_tol():
    """max_tol に負の値を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            max_tol=-10, # 型はあっているが、値の範囲が無効
        )
    assert "max_tol must be a non-negative float" in str(excinfo.value)

def test_kmodes_invalid_type_max_tol():
    """max_tol に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            max_tol="10",  # 型があっていない
        )
    assert "max_tol must be a non-negative float" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_max_tol():
    """max_tol に整数または浮動小数点の正の値を指定した場合に正常動作するかをテストする。"""
    model = FasterKPrototypes(
            max_tol=10, # 正常な型と範囲
         )
    assert model.max_tol == 10

    model = FasterKPrototypes(
            max_tol=10.0, # 正常な型と範囲
         )
    assert model.max_tol == 10.0

def test_kprototypes_invalid_range_max_tol():
    """max_tol に負の値を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            max_tol=-10, # 型はあっているが、値の範囲が無効
        )
    assert "max_tol must be a non-negative float" in str(excinfo.value)

def test_kprototypes_invalid_type_max_tol():
    """max_tol に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            max_tol="10",  # 型があっていない
        )
    assert "max_tol must be a non-negative float" in str(excinfo.value)
