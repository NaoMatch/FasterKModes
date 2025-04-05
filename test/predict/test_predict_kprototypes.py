import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKPrototypes import FasterKPrototypes
from valid_hyperparameters import KPROTOTYPES_VALID_HYPERPARAMETERS

def test_kprototypes_valid_predict():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    model = FasterKPrototypes(**KPROTOTYPES_VALID_HYPERPARAMETERS)
    model.fit(X, categorical=[1,2,3,5])
    predict = model.predict(X)

    assert model.n_clusters == 2


def test_kprototypes_invalid_non_fit_predict():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KPROTOTYPES_VALID_HYPERPARAMETERS)
        model.predict(X)
    assert "Model has not been fitted. Please call 'fit' before using 'predict'." in str(excinfo.value)


def test_kprototypes_invalid_type_predict():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = [
        [1,2,3]
    ]
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KPROTOTYPES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,2,3,5])
        model.predict(X_test)
    assert "X must be a numpy ndarray." in str(excinfo.value)


def test_kprototypes_invalid_ndim_predict():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.random.randint(0, 256, (2,2,2), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KPROTOTYPES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,2,3,5])
        model.predict(X_test)
    assert "X must be a 2-dimensional array." in str(excinfo.value)


def test_kprototypes_invalid_n_cols_predict():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.random.randint(0, 256, (2,100), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KPROTOTYPES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,2,3,5])
        model.predict(X_test)
    assert "X must have the same number of columns as the training data. " in str(excinfo.value)


def test_kprototypes_invalid_not_acceptable_predict():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.random.randint(0, 256, (10,10), dtype=np.uint32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KPROTOTYPES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,2,3,5])
        model.predict(X_test)
    assert "X must have a dtype of uint8, uint16, float32, or float64." in str(excinfo.value)


def test_kprototypes_invalid_negative_category_predict():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.random.randint(-256, 256, (10,10)).astype(np.float32)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KPROTOTYPES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,2,3,5])
        model.predict(X_test)
    assert "Categorical features in X must be non-negative." in str(excinfo.value)


def test_kprototypes_invalid_categorical_over_range_predict():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.zeros((10,10))
    X_test[:,0] += np.finfo(np.float64).max
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KPROTOTYPES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,2,3,5])
        model.predict(X_test)
    assert "Numerical features in X must not exceed " in str(excinfo.value)

def test_kprototypes_invalid_order_predict():
    """
    有効な文字列 init を指定した場合、FasterKPrototypes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.array(np.zeros((10,10)), order="F")
    with pytest.raises(ValueError) as excinfo:
        model = FasterKPrototypes(**KPROTOTYPES_VALID_HYPERPARAMETERS)
        model.fit(X, categorical=[1,2,3,5])
        model.predict(X_test)
    assert "X must have C-order. Ensure the array is row-major memory layout." in str(excinfo.value)
