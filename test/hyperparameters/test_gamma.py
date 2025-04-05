import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# FasterKPrototypes 用テスト
def test_kprototypes_valid_gamma():
    """gamma に整数または浮動小数点の正の値を指定した場合に正常動作するかをテストする。"""
    model = FasterKPrototypes(
            gamma=10, # 正常な型と範囲
         )
    assert model.gamma == 10

    model = FasterKPrototypes(
            gamma=10.0, # 正常な型と範囲
         )
    assert model.gamma == 10.0

def test_kprototypes_invalid_range_gamma():
    """gamma に負の値を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            gamma=-10, # 型はあっているが、値の範囲が無効
        )
    assert "gamma must be a non-negative int or float (or None)" in str(excinfo.value)

def test_kprototypes_invalid_type_gamma():
    """gamma に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            gamma="10",  # 型があっていない
        )
    assert "gamma must be a non-negative int or float (or None)" in str(excinfo.value)
