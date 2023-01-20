"""
utils.py is a module for helper functions in the dw_events.data subpackage.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
import datetime

def make_dt_list(
    start_dt: datetime.datetime,
    stop_dt: datetime.datetime,
    interval=60):
    """Generates a list of datetime timestamps
    between a start_dt and a stop_dt

    Args:
        start_dt (datetime.datetime): Starting timestamp of the list
        stop_dt (datetime.datetime): Final timestamp of the list
        interval (int, optional): Interval in seconds between each step.
            Defaults to 60.

    Returns:
        _type_: _description_
    """
    nr_steps = int((stop_dt - start_dt).total_seconds() / interval)
    dt_list = [start_dt + datetime.timedelta(0, interval * x) for x in range(nr_steps)]
    return dt_list
