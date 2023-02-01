"""
utils.py is a module for helper functions in the
dw_events.temperature_compensation subpackage.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from math import *


def gaussian_function(
    mean: float,
    var: float,
    value: float,
) -> float:
    """The function takes in
    a mean and squared variance, and an input x
    and returns the gaussian value.

    Args:
        mean (float): The mean of the gaussian function.
        var (float): The variance of the gaussian function.
        value (float): The input value.

    Returns:
        float: The gaussian value.
    """
    variance_squared = var**2
    coefficient = 1.0 / sqrt(2.0 * pi * variance_squared)
    exponential = exp(-0.5 * (value - mean) ** 2 / variance_squared)
    return coefficient * exponential


def gaussian_update(
    mean1: float,
    var1: float,
    mean2: float,
    var2: float,
) -> list[float]:
    """This function takes in two means and two squared variance terms,
    and returns updated gaussian parameters.

    Args:
        mean1 (float): The initial mean.
        var1 (float): The initial variance squared.
        mean2 (float): The mean used for the update.
        var2 (float): The variance used for the update squared.

    Returns:
        _type_: _description_
    """
    new_mean = (var2 * mean1 + var1 * mean2) / (var2 + var1)
    new_var = 1 / (1 / var2 + 1 / var1)
    return [new_mean, new_var]
