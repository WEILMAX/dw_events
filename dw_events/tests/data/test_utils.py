"""
Test of the utils module.
"""
import datetime
from dw_events.data.utils import make_dt_list


def test_make_dt_list():
    """Test make_dt_list()."""
    start_dt = datetime.datetime(2022, 1, 1, 0, 0, 0)
    stop_dt = datetime.datetime(2022, 1, 1, 1, 0, 0)
    interval = 600
    dt_list = make_dt_list(start_dt, stop_dt, interval)
    assert len(dt_list) == 6
    assert dt_list[0] == start_dt
    assert dt_list[-1] == stop_dt - datetime.timedelta(0, interval)
    assert dt_list[1] - dt_list[0] == datetime.timedelta(0, interval)


def test_make_dt_list_default_interval():
    """Test make_dt_list() with default interval."""
    start_dt = datetime.datetime(2022, 1, 1, 0, 0, 0)
    stop_dt = datetime.datetime(2022, 1, 1, 1, 0, 0)
    dt_list = make_dt_list(start_dt, stop_dt)
    assert len(dt_list) == 60
    assert dt_list[0] == start_dt
    assert dt_list[-1] == stop_dt - datetime.timedelta(0, 60)
    assert dt_list[1] - dt_list[0] == datetime.timedelta(0, 60)
