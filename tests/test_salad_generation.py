import pytest
from pandas import DataFrame
from pandas.api import types as ptypes
from typing import Dict, List
from datetime import datetime

from src.feature_salad import FeatureSalad


def feature_df(features: List[Dict], samples: int) -> DataFrame:
    fs = FeatureSalad(features=features, samples=samples)
    fs.generate()
    return fs.X

@pytest.fixture
def samples() -> int:
    return 100

@pytest.fixture
def datetime_feature() -> Dict:
    return {
        'n': 1,
        'dtype': 'datetime',
        'between': ['2022-01-01', '2022-12-31']
    }

@pytest.fixture
def category_feature() -> Dict:
    return {
        'n': 1,
        'dtype': 'category',
        'distinct': 8
    }

@pytest.fixture
def category_integers_feature() -> Dict:
    return {
        'n': 1,
        'dtype': 'category',
        'made_of': 'integers',
        'distinct': 8
    }

@pytest.fixture
def boolean_feature() -> Dict:
    return {
        'n': 1,
        'dtype': 'boolean'
    }

@pytest.fixture
def float_feature() -> Dict:
    return {
        'n': 1,
        'dtype': 'float',
        'between': [0.0, 1.0]
    }

@pytest.fixture
def int_feature() -> Dict:
    return {
        'n': 1,
        'dtype': 'int',
        'between': [5, 20]
    }

@pytest.fixture
def datetime_df(datetime_feature, samples) -> DataFrame:
    return feature_df(
        features=[datetime_feature],
        samples=samples
    )

@pytest.fixture
def category_df(category_feature, samples) -> DataFrame:
    return feature_df(
        features=[category_feature],
        samples=samples
    )

@pytest.fixture
def category_integers_df(category_integers_feature, samples) -> DataFrame:
    return feature_df(
        features=[category_integers_feature],
        samples=samples
    )

@pytest.fixture
def boolean_df(boolean_feature, samples) -> DataFrame:
    return feature_df(
        features=[boolean_feature],
        samples=samples
    )

@pytest.fixture
def float_df(float_feature, samples) -> DataFrame:
    return feature_df(
        features=[float_feature],
        samples=samples
    )

@pytest.fixture
def int_df(int_feature, samples) -> DataFrame:
    return feature_df(
        features=[int_feature],
        samples=samples
    )


def test_datetime_df(datetime_df, datetime_feature) -> None:
    assert ptypes.is_datetime64_any_dtype(datetime_df.iloc[:, 0])
    assert datetime_df.iloc[:, 0].min() >= datetime.fromisoformat(datetime_feature['between'][0]), 'dates out of range.'
    assert datetime_df.iloc[:, 0].max() <= datetime.fromisoformat(datetime_feature['between'][1]), 'dates out of range.'

def test_category_df(category_df, category_feature) -> None:
    assert ptypes.is_categorical_dtype(category_df.iloc[:, 0])
    assert len(category_df.iloc[:, 0].unique()) == category_feature['distinct'], 'distinct values mismatch.'

def test_category_integers_df(category_integers_df) -> None:
    assert ptypes.is_categorical_dtype(category_integers_df.iloc[:, 0])
    assert int(category_integers_df.iloc[:, 0][0]), 'not made of integers.'

def test_boolean_df(boolean_df, samples) -> None:
    assert ptypes.is_bool_dtype(boolean_df.iloc[:, 0])
    assert len(boolean_df.index) == samples, 'wrong number of samples.'

def test_float_df(float_df, float_feature) -> None:
    assert ptypes.is_float_dtype(float_df.iloc[:, 0])
    assert float_df.iloc[:, 0].min() >= float_feature['between'][0], 'values out of range.'
    assert float_df.iloc[:, 0].max() <= float_feature['between'][1], 'values out of range.'

def test_int_df(int_df, int_feature) -> None:
    assert ptypes.is_integer_dtype(int_df.iloc[:, 0])
    assert int_df.iloc[:, 0].min() >= int_feature['between'][0], 'values out of range.'
    assert int_df.iloc[:, 0].max() <= int_feature['between'][1], 'values out of range.'
