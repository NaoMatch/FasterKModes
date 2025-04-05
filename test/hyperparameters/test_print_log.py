import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from FasterKModes import FasterKModes

# FasterKModes 用テスト

def test_kmodes_valid_print_log():
    """print_log に True を指定した場合に正常動作するかをテストする。"""
    model = FasterKModes(
            print_log=True, # 正常な型
         )
    assert model.print_log == True

def test_kmodes_invalid_type_print_log():
    """print_log に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            print_log="10",  # 型があっていない
        )
    assert "print_log must be a boolean value, but got" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_print_log():
    """print_log に True を指定した場合に正常動作するかをテストする。"""
    model = FasterKPrototypes(
            print_log=True, # 正常な型
         )
    assert model.print_log == True

def test_kprototypes_invalid_type_print_log():
    """print_log に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            print_log="10",  # 型があっていない
        )
    assert "print_log must be a boolean value, but got" in str(excinfo.value)
