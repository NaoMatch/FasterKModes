import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKModes import FasterKModes, FasterKPrototypes

# FasterKModes 用テスト

@pytest.mark.parametrize("cpu_count", [2, 16, None])
def test_kmodes_valid_n_jobs(monkeypatch, cpu_count):
    """明示した n_jobs は CPU 数にかかわらず保持する。"""
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    model = FasterKModes(
            n_jobs=10, # 正常な型と範囲
         )
    assert model.n_jobs == 10

def test_kmodes_invalid_range_n_jobs():
    """n_jobs に 0 以下の値を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_jobs=0, # 型はあっているが、値の範囲が無効
        )
    assert "n_jobs" in str(excinfo.value)

def test_kmodes_invalid_type_n_jobs():
    """n_jobs に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKModes(
            n_jobs="10",  # 型があっていない
        )
    assert "n_jobs" in str(excinfo.value)


# FasterKPrototypes 用テスト

@pytest.mark.parametrize("cpu_count", [2, 16, None])
def test_kprototypes_valid_n_jobs(monkeypatch, cpu_count):
    """明示した n_jobs は CPU 数にかかわらず保持する。"""
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    model = FasterKPrototypes(
            n_jobs=10, # 正常な型と範囲
         )
    assert model.n_jobs == 10

def test_kprototypes_invalid_range_n_jobs():
    """n_jobs に 0 以下の値を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_jobs=0, # 型はあっているが、値の範囲が無効
        )
    assert "n_jobs" in str(excinfo.value)

def test_kprototypes_invalid_type_n_jobs():
    """n_jobs に文字列など無効な型を指定した場合に ValueError が発生するかをテストする。"""
    with pytest.raises(ValueError) as excinfo:
        FasterKPrototypes(
            n_jobs="10",  # 型があっていない
        )
    assert "n_jobs" in str(excinfo.value)


@pytest.mark.parametrize("model_class", [FasterKModes, FasterKPrototypes])
@pytest.mark.parametrize("cpu_count", [1, 4])
def test_none_n_jobs_uses_cpu_count(monkeypatch, model_class, cpu_count):
    """n_jobs=None は CPU 数を使用する。"""
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    assert model_class(n_jobs=None).n_jobs == cpu_count
