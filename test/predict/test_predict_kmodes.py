import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
from FasterKModes import FasterKModes, FasterKPrototypes
from valid_hyperparameters import KMODES_VALID_HYPERPARAMETERS

def test_kmodes_valid_predict():
    """
    有効な文字列 init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
    model.fit(X)
    predict = model.predict(X)

    assert model.n_clusters == 2


def test_kmodes_invalid_non_fit_predict():
    """
    有効な文字列 init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.predict(X)
    assert "Error: Model has not been fitted. Please call 'fit' before using 'predict'." in str(excinfo.value)


def test_kmodes_invalid_type_predict():
    """
    有効な文字列 init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = [
        [1,2,3]
    ]
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
        model.predict(X_test)
    assert "Error: X must be a numpy array. Current input type: " in str(excinfo.value)


def test_kmodes_invalid_ndim_predict():
    """
    有効な文字列 init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.random.randint(0, 256, (2,2,2), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
        model.predict(X_test)
    assert "Error: X must be a 2D array. Current ndim: " in str(excinfo.value)


def test_kmodes_invalid_dtype_mismatch_predict():
    """
    有効な文字列 init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.random.randint(0, 256, (10,10), dtype=np.uint16)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
        model.predict(X_test)
    assert "Error: X's dtype " in str(excinfo.value)


def test_kmodes_invalid_order_predict():
    """
    有効な文字列 init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.random.randint(0, 256, (10,10), dtype=np.uint8)
    X_test = np.array(X_test, order="F")
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
        model.predict(X_test)
    assert "Error: X must be C-order (row-major memory layout)" in str(excinfo.value)


def test_kmodes_invalid_n_cols_predict():
    """
    有効な文字列 init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.random.randint(0, 256, (10,100), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
        model.predict(X_test)
    assert "Error: X must have the same number of columns as the training data. " in str(excinfo.value)


def test_kmodes_invalid_empty_predict():
    """
    有効な文字列 init を指定した場合、FasterKModes インスタンスが正しく作成されることを確認する。
    """
    X = np.random.randint(0, 256, (100, 10), dtype=np.uint8)
    X_test = np.random.randint(0, 256, (0,10), dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        model = FasterKModes(**KMODES_VALID_HYPERPARAMETERS)
        model.fit(X)
        model.predict(X_test)
    assert "Error: X must have at least one row. Got " in str(excinfo.value)

