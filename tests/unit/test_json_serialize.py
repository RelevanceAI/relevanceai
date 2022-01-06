import pytest
import random


def test_datetime_upload(test_datetime_dataset):
    response, original_length = test_datetime_dataset
    assert response["inserted"] == original_length


def test_numpy_upload(test_numpy_dataset):
    response, original_length = test_numpy_dataset
    assert response["inserted"] == original_length


def test_pandas_upload(test_pandas_dataset):
    response, original_length = test_pandas_dataset
    assert response["inserted"] == original_length


def test_nested_assorted_upload(test_nested_assorted_dataset):
    response, original_length = test_nested_assorted_dataset
    assert response["inserted"] == original_length


def test_csv_upload(test_csv_dataset):
    response, original_length = test_csv_dataset
    assert response["inserted"] == original_length
