"""
savgol_filter.py is a module for applying the savgol filter
as a temperature compensation method.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field
from typing import Union

import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class SavgolTempComp:
    """
    A class for applying the Savitzky-Golay filter as a temperature compensation method.
    The Savitzky-Golay filter is a type of low-pass filter that is commonly used to smooth data.

    Attributes:
        data (pd.DataFrame): DataFrame containing the data to be filtered.
        window_length (int, optional): The window length to use for the filter in seconds.
            Defaults to 3.
        polyorder (int, optional): The polynomial order to use for the filter.
            Defaults to 1.
    """

    data: pd.DataFrame
    window_length: int = 3
    polyorder: int = 1

    def __post_init__(self):
        self.filtered_data: pd.DataFrame = field(init=False)
        self.compensated_data: pd.DataFrame = field(init=False)

    def resample_data(self, freq: str = "1s") -> pd.DataFrame:
        """
        Resample the data to the specified frequency.

        Args:
            freq (str, optional): The frequency to resample the data to.
                Defaults to "1s"

        Returns:
            pd.DataFrame: Resampled data
        """
        resampled_data = self.data.copy()
        return resampled_data.resample(freq).mean()

    def filter_data(
        self, freq: str = "1s", min_max: str = "", min_max_freq: int = 30
    ) -> Union[pd.DataFrame, None]:
        """
        Apply the Savitzky-Golay filter to the data.

        Args:
            freq (str): The frequency to resample the data to before filtering.
            min_max (str, optional): The min or max value to use for resampling.
                Determines if we want to follow the upper or lower limit of the signal.
                If the sensor is loaded in tension == max, compression == min.
                Defaults to ''.
            min_max_freq (int, optional): The frequency to resample the data to in seconds,
                before filtering with min_max.
                Defaults to 30.

        Returns:
            pd.DataFrame: Filtered data
        """
        resampled_data = self.resample_data(freq)
        self.filtered_data = resampled_data.copy()
        window_length = self.window_length
        if min_max == "min":
            self.filtered_data = self.filtered_data.resample(f"{min_max_freq}s").min()
            window_length = int(self.window_length / min_max_freq)
        elif min_max == "max":
            self.filtered_data = self.filtered_data.resample(f"{min_max_freq}s").max()
            window_length = int(self.window_length / min_max_freq)
        for column in self.data.columns:
            self.filtered_data[column] = savgol_filter(
                self.filtered_data[column], window_length, self.polyorder
            )
        self.filtered_data = self.filtered_data.reindex(
            self.data.index
        ).interpolate()  # type: ignore
        return self.filtered_data

    def apply_filter(
        self, freq: str = "1s", min_max: str = "", min_max_freq: int = 30
    ) -> pd.DataFrame:
        """
        Use the Savitzky-Golay filtered data to compensate for temperature effects.

        Args:
            freq (str): The frequency to resample the data to before filtering.
            min_max (str, optional): The min or max value to use for resampling.
                Determines if we want to follow the upper or lower limit of the signal.
                If the sensor is loaded in tension == max, compression == min.
                Defaults to ''.
            min_max_freq (int, optional): The frequency to resample the data to in seconds,
                before filtering with min_max.
                Defaults to 30.

        Returns:
            pd.DataFrame: Filtered data
        """
        self.filter_data(freq, min_max, min_max_freq)
        self.compensated_data = self.data - self.filtered_data
        return self.compensated_data
