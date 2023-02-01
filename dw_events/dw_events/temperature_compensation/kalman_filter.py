"""
kalman_filter.py is a module for applying the kalman filter
as a temperature compensation method.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class KalmanFilter1D:
    """
    A simple Kalman filter implementation for 1D data smoothing.

    Attributes:
    process_variance (float): The variance of the process noise,
        which represents the uncertainty of the system's state transition model (=Q).
    estimated_measurement_variance (float): The variance of the measurement noise,
        which represents the uncertainty of the measurement model (=R).
    posteri_estimate (float, optional): The latest estimated measurement,
        initialized to 0.0.
    posteri_error_estimate (float, optional): The latest estimated error of the measurement,
        initialized to 1.0.
    """

    process_variance: float  # Q
    estimated_measurement_variance: float  # R
    posteri_estimate:float = 0.0  # x^_0
    posteri_error_estimate:float = 1.0  # P_0

    def input_latest_noisy_measurement(
        self,
        measurement: float,
        input: float = 0.0,
        A_matrix: float = 1.0,
        B_matrix: float = 1.0):
        """Input the latest noisy measurement and
        update the posteri estimate and posteri error estimate.

        Args:
        measurement (float): The latest noisy measurement (=y_k).
        """
        ## Time Update (Prediction)
        # x_hat_predicted = A * x_hat_posteri + B * u
        priori_estimate = A_matrix * self.posteri_estimate + B_matrix * input
        # P_predicted = A * P_posteri * A.T + Q
        priori_error_estimate = \
            A_matrix * self.posteri_error_estimate * A_matrix\
            + self.process_variance

        ## Measurement Update (Correction)
        # K = P_predicted * H.T * (H * P_predicted * H.T + R).inverse()
        blending_factor = priori_error_estimate / (
            priori_error_estimate + self.estimated_measurement_variance
        )
        # x_hat_posteri = x_hat_predicted + K * (z - H * x_hat_predicted)
        self.posteri_estimate = priori_estimate + blending_factor * (
            measurement - priori_estimate
        )
        # P_posteri = (I - K * H) * P_predicted
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

    def get_latest_estimated_measurement(self) -> float:
        """
        Get the latest estimated measurement.

        Returns:
        float: The latest estimated measurement.
        """
        return self.posteri_estimate

    def filter_data(self, strain_data: pd.Series) -> pd.Series:
        """Apply the kalman filter to the strain data.

        Args:
        strain_data (pd.Series): The strain data to apply the kalman filter to.

        Returns:
        pd.Series: The filtered strain data.
        """
        filtered_data_values = []
        for index, value in strain_data.iteritems():
            self.input_latest_noisy_measurement(value)
            filtered_data_values.append(
                self.get_latest_estimated_measurement()
            )
        filtered_data = \
            pd.Series(
                filtered_data_values,
                index=strain_data.index,
                dtype=float
            )
        return filtered_data
    
    def apply_filter(
        self,
        strain_data: pd.Series
        ) -> pd.Series:
        """Apply the kalman filter for
        temperature compensation of the strain data.

        Args:
        strain_data (pd.Series): The strain data to apply the kalman filter to.

        Returns:
        pd.Series: The filtered strain data.
        """
        filtered_data = self.filter_data(strain_data)
        compensated_data = strain_data - filtered_data
        return compensated_data
