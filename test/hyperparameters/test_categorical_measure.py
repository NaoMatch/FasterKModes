import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes
from valid_hyperparameters import custom_valid_args_categorical_measure, custom_invalid_args_categorical_measure
from config import VALID_CATEGORICAL_MEASURES

# FasterKModes 用テスト

def test_kmodes_valid_str_categorical_measure():
    """
    有効な文字列 categorical_measure を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    for categorical_measure in VALID_CATEGORICAL_MEASURES:
        model = FasterKModes(
                categorical_measure=categorical_measure, # 正常なstr
            )
        assert model.categorical_measure == categorical_measure

def test_kmodes_invalid_str_categorical_measure():
    """
    無効な文字列 categorical_measure を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
                categorical_measure="categorical_measure", # 不正なstr
            )
    assert "categorical_measure must be one of" in str(excinfo.value)

def test_kmodes_invalid_int_categorical_measure():
    """
    categorical_measure に整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
                categorical_measure=12, # 不正な型
            )
    assert "categorical_measure must be a string from" in str(excinfo.value)

def test_kmodes_valid_args_callable_categorical_measure():
    """
    正しい引数を持つ callable な categorical_measure を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKModes(
            categorical_measure=custom_valid_args_categorical_measure, # 正常なcallable
        )
    assert model.categorical_measure == custom_valid_args_categorical_measure

def test_kmodes_invalid_args_callable_categorical_measure():
    """
    引数が正しくない callable を categorical_measure に指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
                categorical_measure=custom_invalid_args_categorical_measure, # 不正なcallable
            )
    assert "Custom categorical_measure function must accept exactly two arguments: 'x_cat' and 'c_cat'." in str(excinfo.value)

# FasterKPrototypes 用テスト

def test_kprototypes_valid_str_categorical_measure():
    """
    有効な文字列 categorical_measure を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    for categorical_measure in VALID_CATEGORICAL_MEASURES:
        model = FasterKPrototypes(
                categorical_measure=categorical_measure, # 正常なstr
            )
        assert model.categorical_measure == categorical_measure

def test_kprototypes_invalid_str_categorical_measure():
    """
    無効な文字列 categorical_measure を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            categorical_measure="categorical_measure", # 不正なstr
            )
    assert "categorical_measure must be one of" in str(excinfo.value)

def test_kprototypes_invalid_int_categorical_measure():
    """
    categorical_measure に整数を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            categorical_measure=12, # 不正な型
            )
    assert "categorical_measure must be a string from" in str(excinfo.value)

def test_kprototypes_valid_args_callable_categorical_measure():
    """
    正しい引数を持つ callable な categorical_measure を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKPrototypes(
            categorical_measure=custom_valid_args_categorical_measure, # 正常なcallable
        )
    assert model.categorical_measure == custom_valid_args_categorical_measure

def test_kprototypes_invalid_args_callable_categorical_measure():
    """
    引数が正しくない callable を categorical_measure に指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            categorical_measure=custom_invalid_args_categorical_measure, # 不正なcallable
            )
    assert "Custom categorical_measure function must accept exactly two arguments: 'x_cat' and 'c_cat'." in str(excinfo.value)
