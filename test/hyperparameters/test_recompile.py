import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKModes import FasterKModes, FasterKPrototypes

# FasterKModes 用テスト

def test_kmodes_valid_recompile():
    """recompile に True を指定した場合に正常動作するかをテストする。"""
    model = FasterKModes(
            recompile=True, # 正常な型
         )
    assert model.recompile == True

def test_kmodes_invalid_type_recompile():
    """recompile に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            recompile="10",  # 型があっていない
        )
    assert "recompile must be a boolean value, but got" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_recompile():
    """recompile に True を指定した場合に正常動作するかをテストする。"""
    model = FasterKPrototypes(
            recompile=True, # 正常な型
         )
    assert model.recompile == True

def test_kprototypes_invalid_type_recompile():
    """recompile に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            recompile="10",  # 型があっていない
        )
    assert "recompile must be a boolean value, but got" in str(excinfo.value)
