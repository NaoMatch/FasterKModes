import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# FasterKModes 用テスト

def test_kmodes_valid_max_iter():
    """
    max_iter に正の整数を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKModes(
            max_iter=10, # 正常な型と範囲
         )
    assert model.max_iter == 10

def test_kmodes_invalid_range_max_iter():
    """
    max_iter に 0 以下の値を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            max_iter=0, # 型はあっているが、値の範囲が無効
        )
    assert "max_iter" in str(excinfo.value)

def test_kmodes_invalid_type_max_iter():
    """
    max_iter に整数以外の型（文字列）を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            max_iter="10",  # 型があっていない
        )
    assert "max_iter" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_max_iter():
    """
    max_iter に正の整数を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKPrototypes(
            max_iter=10, # 正常な型と範囲
         )
    assert model.max_iter == 10

def test_kprototypes_invalid_range_max_iter():
    """
    max_iter に 0 以下の値を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            max_iter=0, # 型はあっているが、値の範囲が無効
        )
    assert "max_iter" in str(excinfo.value)

def test_kprototypes_invalid_type_max_iter():
    """
    max_iter に整数以外の型（文字列）を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            max_iter="10",  # 型があっていない
        )
    assert "max_iter" in str(excinfo.value)
