import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# FasterKModes 用テスト

def test_kmodes_valid_none_n_init():
    """
    random_state に None を指定した場合、FasterKModes のインスタンスが正しく作成されることを確認する。
    """
    model = FasterKModes(
                random_state=None, # 正常な型と範囲
         )
    assert model.random_state is None

def test_kmodes_valid_int_n_init():
    """
    random_state に正の整数を指定した場合、FasterKModes のインスタンスが正しく作成されることを確認する。
    """
    model = FasterKModes(
                random_state=10, # 正常な型と範囲
         )
    assert model.random_state is 10

def test_kmodes_invalid_range_negative_int_n_init():
    """
    random_state に負の整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
                random_state=-10, # 型はあっているが、値の範囲が無効
            )
    assert "random_state must be a non-negative integer" in str(excinfo.value)

def test_kmodes_invalid_range_huge_int_n_init():
    """
    random_state に許容範囲を超える大きな整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
                random_state=pow(2,32), # 型はあっているが、値の範囲が無効
            )
    assert "random_state must be less than or equal" in str(excinfo.value)

def test_kmodes_invalid_type_int_n_init():
    """
    random_state に文字列型の "None" を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
                random_state="None", # 型があっていない
            )
    assert "random_state must be either None or an integer" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_none_n_init():
    """
    random_state に None を指定した場合、FasterKPrototypes のインスタンスが正しく作成されることを確認する。
    """
    model = FasterKPrototypes(
                random_state=None, # 正常な型と範囲
         )
    assert model.random_state is None

def test_kprototypes_valid_int_n_init():
    """
    random_state に正の整数を指定した場合、FasterKPrototypes のインスタンスが正しく作成されることを確認する。
    """
    model = FasterKPrototypes(
                random_state=10, # 正常な型と範囲
         )
    assert model.random_state is 10

def test_kprototypes_invalid_range_negative_int_n_init():
    """
    random_state に負の整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
                random_state=-10, # 型はあっているが、値の範囲が無効
            )
    assert "random_state must be a non-negative integer" in str(excinfo.value)

def test_kprototypes_invalid_range_huge_int_n_init():
    """
    random_state に許容範囲を超える大きな整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
                random_state=pow(2,32), # 型はあっているが、値の範囲が無効
            )
    assert "random_state must be less than or equal" in str(excinfo.value)

def test_kprototypes_invalid_type_int_n_init():
    """
    random_state に文字列型の "None" を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
                random_state="None", # 型があっていない
            )
    assert "random_state must be either None or an integer" in str(excinfo.value)
