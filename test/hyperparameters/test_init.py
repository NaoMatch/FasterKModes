import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes
from valid_hyperparameters import custom_valid_init_kmodes, custom_valid_init_kprototypes
from valid_hyperparameters import custom_invalid_args_init_kmodes, custom_invalid_args_init_kprototypes
from config import VALID_INIT_METHODS

# FasterKModes 用テスト

def test_kmodes_valid_str_init():
    """
    有効な文字列 init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    for init_method in VALID_INIT_METHODS:
        model = FasterKModes(
                init=init_method, # 正常なstr
            )
        assert model.init == init_method

def test_kmodes_invalid_str_init():
    """
    無効な文字列 init を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
                init="lloyd", # 不正なstr
            )
    assert "init must be one of" in str(excinfo.value)

def test_kmodes_invalid_int_init():
    """
    init に整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
                init=12, # 不正な型
            )
    assert "init must be a string or a callable function" in str(excinfo.value)

def test_kmodes_valid_callable_init():
    """
    正しい引数を持つ callable な init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKModes(
            init=custom_valid_init_kmodes, # 正常なcallable
        )
    assert model.init == custom_valid_init_kmodes

def test_kmodes_invalid_callable_args_init():
    """
    引数が正しくない callable を init に指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
                init=custom_invalid_args_init_kmodes, # 不正なcallable
            )
    assert "Custom init function must accept exactly two arguments: 'X' and 'n_clusters'" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_str_init():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    for init_method in VALID_INIT_METHODS:
        model = FasterKPrototypes(
                init=init_method,  # 正常なstr
            )
        assert model.init == init_method

def test_kprototypes_invalid_str_init():
    """
    無効な文字列 init を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
                init="lloyd", # 不正なstr
            )
    assert "init must be one of" in str(excinfo.value)

def test_kprototypes_invalid_int_init():
    """
    init に整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
                init=12, # 不正な型
            )
    assert "init must be a string or a callable function" in str(excinfo.value)

def test_kprototypes_valid_callable_init():
    """
    正しい引数を持つ callable な init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKPrototypes(
            init=custom_valid_init_kprototypes,  # 正常なcallable
        )
    assert model.init == custom_valid_init_kprototypes

def test_kprototypes_invalid_callable_args_init():
    """
    引数が正しくない callable を init に指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
                init=custom_invalid_args_init_kprototypes, # 不正なcallable
            )
    assert "Custom init function must accept exactly two arguments: 'Xcat', 'Xnum', and 'n_clusters'. Got parameters: " in str(excinfo.value)
