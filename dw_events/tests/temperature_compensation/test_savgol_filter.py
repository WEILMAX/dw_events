"""
Test of the savgol_filter module.
"""

import datetime
import pandas as pd
from dw_events.temperature_compensation.savgol_filter import SavgolTempComp

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dw_events.temperature_compensation.savgol_filter import SavgolTempComp


@pytest.fixture
def test_data():
    """
    Create a dataset for the tests.
    """
    data = {
        "data1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "data2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    index = [datetime.now() + timedelta(seconds=i) for i in range(10)]  # type: ignore
    return pd.DataFrame(data, index=index)


def test_resample_data(test_data):
    """
    Test data resampling.
    """
    savgol = SavgolTempComp(test_data)
    resampled_data = savgol.resample_data(freq="1s")
    assert resampled_data.index.freq == "1s"
    assert len(resampled_data) == 10


def test_filter_data(test_data):
    """
    Test applying savgol filter.
    """
    savgol = SavgolTempComp(test_data)
    filtered_data = savgol.filter_data(freq="1s")
    assert len(filtered_data) == 10
    for column in test_data.columns:
        assert (
            filtered_data[column].values.tolist() != test_data[column].values.tolist()
        )


def test_apply_filter(test_data):
    """
    Test applying the savgol temperature compensation.
    """
    savgol = SavgolTempComp(test_data)
    compensated_data = savgol.apply_filter(freq="1s")
    assert len(compensated_data) == 10
    for column in test_data.columns:
        assert (
            compensated_data[column].values.tolist()
            != test_data[column].values.tolist()
        )
