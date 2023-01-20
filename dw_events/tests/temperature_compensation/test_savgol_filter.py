"""
Test of the savgol_filter module.
"""

import datetime
import pandas as pd
from dw_events.temperature_compensation.savgol_filter import SavgolTempComp


def test_resample_data():
    """
    Test resampling of data
    """
    # Create a test DataFrame
    data = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6]},
        index=[
            datetime.datetime(2020, 12, 31, 23, 59, 50, 0),
            datetime.datetime(2020, 12, 31, 23, 59, 50, 50),
            datetime.datetime(2020, 12, 31, 23, 59, 51, 0),
            datetime.datetime(2020, 12, 31, 23, 59, 51, 50),
            datetime.datetime(2020, 12, 31, 23, 59, 52, 0),
        ],
    )
    # Create an instance of the SavgolTempComp class
    temp_comp = SavgolTempComp(data)
    # Test the resample_data() method
    temp_comp.resample_data(freq="1s")
    assert temp_comp.data.index.freq == "S"  # type: ignore


def test_apply_filter():
    """
    Test applying savgol filter
    """
    data = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6]},
        index=[
            datetime.datetime(2020, 12, 31, 23, 59, 50),
            datetime.datetime(2020, 12, 31, 23, 59, 51),
            datetime.datetime(2020, 12, 31, 23, 59, 52),
            datetime.datetime(2020, 12, 31, 23, 59, 53),
            datetime.datetime(2020, 12, 31, 23, 59, 54),
        ],
    )
    temp_comp = SavgolTempComp(data)
    filtered_data = temp_comp.apply_filter()
    # check if filter is applied
    assert (filtered_data != data).any().any()
    # check if filtered data has same columns as original data
    assert (filtered_data.columns == data.columns).all()
    # check if filtered data has same index as original data
    assert (filtered_data.index == data.index).all()


def test_default_window_length_and_polyorder():
    """
    Test if the default window_length and polyorder are as expected
    """
    data = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6]},
        index=[
            datetime.datetime(2020, 12, 31, 23, 59, 50),
            datetime.datetime(2020, 12, 31, 23, 59, 51),
            datetime.datetime(2020, 12, 31, 23, 59, 52),
            datetime.datetime(2020, 12, 31, 23, 59, 53),
            datetime.datetime(2020, 12, 31, 23, 59, 54),
        ],
    )
    temp_comp = SavgolTempComp(data)
    assert temp_comp.window_length == 3
    assert temp_comp.polyorder == 1
