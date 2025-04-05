import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# FasterKModes 用テスト

def test_kmodes_valid_n_init():
    """
    n_init に 1 を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKModes(
            n_init=1, # 正常な型と範囲
         )
    assert model.n_init == 1

def test_kmodes_invalid_range_n_init():
    """
    n_init に負の整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_init=-1, # 型はあっているが、値の範囲が無効
        )
    assert "n_init" in str(excinfo.value)

def test_kmodes_invalid_type_n_init():
    """
    n_init に文字列 "1" を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_init="1", # 型があっていない
        )
    assert "n_init" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_n_init():
    """
    n_init に 1 を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKPrototypes(
            n_init=1, # 正常な型と範囲
         )
    assert model.n_init == 1

def test_kprototypes_invalid_range_n_init():
    """
    n_init に負の整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_init=-1, # 型はあっているが、値の範囲が無効
        )
    assert "n_init" in str(excinfo.value)

def test_kprototypes_invalid_type_n_init():
    """
    n_init に文字列 "1" を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_init="1", # 型があっていない
        )
    assert "n_init" in str(excinfo.value)
