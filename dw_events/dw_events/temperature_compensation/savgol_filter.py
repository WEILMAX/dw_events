"""
savgol_filter.py is a module for applying the savgol filter
as a temperature compensation method.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field
from scipy.signal import savgol_filter
import pandas as pd


@dataclass
class SavgolTempComp:
    """
    A class for applying the Savitzky-Golay filter as a temperature compensation method.
    The Savitzky-Golay filter is a type of low-pass filter that is commonly used to smooth data.

    Attributes:
        data (pd.DataFrame): DataFrame containing the data to be filtered.
        window_length (int): The window length to use for the filter.
        polyorder (int): The polynomial order to use for the filter.
    """

    data: pd.DataFrame
    window_length: int = 3
    polyorder: int = 1

    def __post_init__(self):
        self.filtered_data: pd.DataFrame = field(init=False)

    def resample_data(self, freq: str = "1s"):
        """
        Resample the data to the specified frequency.

        Args:
            freq (str): The frequency to resample the data to.
        """
        self.data = self.data.resample(freq).mean()

    def apply_filter(self, freq: str = "1s"):
        """
        Apply the Savitzky-Golay filter to the data.

        Args:
            freq (str): The frequency to resample the data to before filtering.

        Returns:
            pd.DataFrame: Filtered data
        """
        self.resample_data(freq)
        self.filtered_data = self.data.copy()
        for column in self.data.columns:
            self.filtered_data[column] = savgol_filter(
                self.data[column], self.window_length, self.polyorder
            )
        return self.filtered_data
