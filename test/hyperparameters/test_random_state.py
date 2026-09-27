import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import pytest
from FasterKModes import FasterKModes, FasterKPrototypes

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


@pytest.fixture
def global_rng_state():
    """Keep these tests isolated, including NumPy's cached Gaussian value."""
    original_state = np.random.get_state()
    np.random.seed(12345)
    np.random.randn()
    state = np.random.get_state()
    try:
        yield state
    finally:
        np.random.set_state(original_state)


@pytest.fixture(params=[FasterKModes, FasterKPrototypes], ids=["kmodes", "kprototypes"])
def clustering_case(request):
    rng = np.random.RandomState(7)
    X = rng.randint(0, 5, size=(60, 4)).astype(np.uint8)
    fit_kwargs = {}
    if request.param is FasterKPrototypes:
        X = np.column_stack((X, rng.randn(60, 2))).astype(np.float32)
        fit_kwargs["categorical"] = [0, 1, 2, 3]
    return request.param, X, fit_kwargs


def _make_model(model_class, **kwargs):
    return model_class(n_clusters=3, n_init=3, n_jobs=1, print_log=False, **kwargs)


def _fit_result(model, X, fit_kwargs):
    # FasterKModes updates supplied centroids in place.
    model.fit(X, **{key: value.copy() for key, value in fit_kwargs.items()})
    names = ("C",) if isinstance(model, FasterKModes) else ("Ccat", "Cnum")
    return [getattr(model, name).copy() for name in names] + [
        model.predict(X), model.best_cost, model.compute_score(X)
    ]


def _assert_same_result(actual, expected):
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_array_equal(actual_value, expected_value)


def _assert_global_rng_unchanged(expected):
    actual = np.random.get_state()
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


@pytest.mark.parametrize("init", ["random", "k-means++", "huang", "cao"])
@pytest.mark.parametrize("random_state", [0, 42])
def test_fit_random_state_is_reproducible(clustering_case, global_rng_state, init, random_state):
    model_class, X, fit_kwargs = clustering_case
    model = _make_model(model_class, init=init, random_state=random_state, max_iter=5)
    expected = _fit_result(model, X, fit_kwargs)

    np.random.seed(98765)
    other = _make_model(model_class, init=init, random_state=random_state, max_iter=5)
    _assert_same_result(_fit_result(other, X, fit_kwargs), expected)

    # Repeated fit on the same instance must restart its local random sequence.
    np.random.randn(17)
    _assert_same_result(_fit_result(model, X, fit_kwargs), expected)


@pytest.mark.parametrize("init", ["random", "k-means++", "huang", "cao"])
@pytest.mark.parametrize("random_state", [None, 42])
def test_fit_preserves_global_rng(clustering_case, global_rng_state, init, random_state):
    model_class, X, fit_kwargs = clustering_case
    model = _make_model(model_class, init=init, random_state=random_state, max_iter=5)
    model.fit(X, **fit_kwargs)
    _assert_global_rng_unchanged(global_rng_state)


@pytest.mark.parametrize("random_state", [None, 42])
def test_empty_cluster_preserves_global_rng(clustering_case, global_rng_state, random_state):
    model_class, X, fit_kwargs = clustering_case
    # Identical initial centroids force all samples into cluster zero.
    if model_class is FasterKModes:
        fit_kwargs["init_C"] = np.repeat(X[:1], 3, axis=0)
    else:
        fit_kwargs["init_Ccat"] = np.repeat(X[:1, :4], 3, axis=0).astype(np.uint8)
        fit_kwargs["init_Cnum"] = np.zeros((3, 2), dtype=np.float32)
    model = _make_model(model_class, random_state=random_state, max_iter=1)
    expected = _fit_result(model, X, fit_kwargs)
    assert not model.count_matrix[1:].any()
    _assert_global_rng_unchanged(global_rng_state)
    if random_state is not None:
        _assert_same_result(_fit_result(model, X, fit_kwargs), expected)
        _assert_global_rng_unchanged(global_rng_state)


@pytest.mark.parametrize("random_state", [None, 42])
def test_failed_fit_preserves_global_rng(clustering_case, global_rng_state, random_state):
    model_class, X, fit_kwargs = clustering_case
    model = _make_model(model_class, random_state=random_state)
    with pytest.raises(ValueError, match="unique rows"):
        model.fit(X[:1], **fit_kwargs)
    _assert_global_rng_unchanged(global_rng_state)
