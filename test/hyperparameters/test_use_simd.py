import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKModes import FasterKModes, FasterKPrototypes

# FasterKModes 用テスト

def test_kmodes_valid_use_simd():
    """use_simd に True を指定した場合に正常動作するかをテストする。"""
    model = FasterKModes(
            use_simd=True, # 正常な型
         )
    assert model.use_simd == True

def test_kmodes_invalid_type_use_simd():
    """use_simd に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            use_simd="10",  # 型があっていない
        )
    assert "use_simd must be a boolean value, but got" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_use_simd():
    """use_simd に True を指定した場合に正常動作するかをテストする。"""
    model = FasterKPrototypes(
            use_simd=True, # 正常な型
         )
    assert model.use_simd == True

def test_kprototypes_invalid_type_use_simd():
    """use_simd に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            use_simd="10",  # 型があっていない
        )
    assert "use_simd must be a boolean value, but got" in str(excinfo.value)
