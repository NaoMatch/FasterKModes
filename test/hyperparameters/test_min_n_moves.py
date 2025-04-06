import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKModes import FasterKModes, FasterKPrototypes


# FasterKModes 用テスト

def test_kmodes_valid_min_n_moves():
    """
    min_n_moves に 0 を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKModes(
            min_n_moves=0, # 正常な型と範囲
         )
    assert model.min_n_moves == 0

def test_kmodes_invalid_range_min_n_moves():
    """
    min_n_moves に負の整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            min_n_moves=-1, # 型はあっているが、値の範囲が無効
        )
    assert "min_n_moves" in str(excinfo.value)

def test_kmodes_invalid_type_min_n_moves():
    """
    min_n_moves に文字列を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            min_n_moves="0", # 型があっていない
        )
    assert "min_n_moves" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_min_n_moves():
    """
    min_n_moves に 0 を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKPrototypes(
            min_n_moves=0, # 正常な型と範囲
         )
    assert model.min_n_moves == 0

def test_kprototypes_invalid_range_min_n_moves():
    """
    min_n_moves に負の整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            min_n_moves=-1, # 型はあっているが、値の範囲が無効
        )
    assert "min_n_moves" in str(excinfo.value)

def test_kprototypes_invalid_type_min_n_moves():
    """
    min_n_moves に文字列を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            min_n_moves="0", # 型があっていない
        )
    assert "min_n_moves" in str(excinfo.value)
