import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKModes import FasterKModes, FasterKPrototypes

# FasterKModes 用テスト

def test_kmodes_valid_n_clusters():
    """
    n_clusters に 2 を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKModes(
            n_clusters=2, # 正常な型と範囲
         )
    assert model.n_clusters == 2

def test_kmodes_invalid_range_n_clusters():
    """
    n_clusters に 1 を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters=1,  # 型はあっているが、値の範囲が無効
        )
    assert "n_clusters" in str(excinfo.value)

def test_kmodes_invalid_type_n_clusters():
    """
    n_clusters に文字列 "1" を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_clusters="1",  # 型があっていない
        )
    assert "n_clusters" in str(excinfo.value)


# FasterKPrototypes 用テスト

def test_kprototypes_valid_n_clusters():
    """
    VALID_HYPERPARAMETERS から取得した適切な値を n_clusters に指定した場合、
    FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    model = FasterKPrototypes(
            n_clusters=2, # 正常な型と範囲
         )
    assert model.n_clusters == 2

def test_kprototypes_invalid_range_n_clusters():
    """
    n_clusters に 1 を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters=1,  # 型はあっているが、値の範囲が無効
        )
    assert "n_clusters" in str(excinfo.value)

def test_kprototypes_invalid_type_n_clusters():
    """
    n_clusters に文字列 "1" を指定した場合、ValueError が発生することを確認する。
    """
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_clusters="1",  # 型があっていない
        )
    assert "n_clusters" in str(excinfo.value)
