import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from valid_hyperparameters import custom_valid_args_numerical_measure, custom_invalid_args_numerical_measure
from config import VALID_NUMERICAL_MEASURES

# FasterKPrototypes 用テスト

def test_kprototypes_valid_str_categorical_measure():
    """
    有効な文字列のnumerical_measureを指定したとき、FasterKPrototypesのインスタンスが正しく作成されることを確認する。
    """
    for numerical_measure in VALID_NUMERICAL_MEASURES:
        model = FasterKPrototypes(
                numerical_measure=numerical_measure, # 正常なstr
            )
        assert model.numerical_measure == numerical_measure

def test_kprototypes_invalid_str_categorical_measure():
    """
    無効な文字列のnumerical_measureを指定したとき、ValueErrorが発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            numerical_measure="numerical_measure", # 不正なstr
            )
    assert "numerical_measure must be one of" in str(excinfo.value)

def test_kprototypes_invalid_int_categorical_measure():
    """
    整数型のnumerical_measureを指定したとき、ValueErrorが発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            numerical_measure=12, # 不正な型
            )
    assert "numerical_measure must be a string from" in str(excinfo.value)

def test_kprototypes_valid_args_callable_categorical_measure():
    """
    正しい引数を持つcallableなnumerical_measureを指定したとき、FasterKPrototypesのインスタンスが正しく作成されることを確認する。
    """
    model = FasterKPrototypes(
            numerical_measure=custom_valid_args_numerical_measure, # 正常なcallable
        )
    assert model.numerical_measure == custom_valid_args_numerical_measure

def test_kprototypes_invalid_args_callable_categorical_measure():
    """
    不正な引数を持つcallableなnumerical_measureを指定したとき、ValueErrorが発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            numerical_measure=custom_invalid_args_numerical_measure, # 不正なcallable
            )
    assert "Custom numerical_measure function must accept exactly two arguments: 'x_num' and 'c_num'." in str(excinfo.value)
