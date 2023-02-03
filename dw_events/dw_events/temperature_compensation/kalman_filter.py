"""
kalman_filter.py is a module for applying the kalman filter
as a temperature compensation method.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import pandas as pd


@dataclass
class KalmanFilter:
    """
    A simple Kalman filter implementation for multidimensional data smoothing.
    """
    process_variance: np.ndarray  # Q
    estimated_measurement_variance: np.ndarray  # R
    posteri_estimate: np.ndarray  # x^_0
    posteri_error_estimate: np.ndarray  # P_0
    filtered_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    a_matrix: np.ndarray = np.empty((0,))
    b_matrix: np.ndarray = np.empty((0,))
    c_matrix: np.ndarray = np.empty((0,))


    def input_latest_noisy_measurement(
        self,
        y_measurement: np.ndarray,
        u_input: Union[np.ndarray, None] = None,
        ):
        """Input the latest noisy measurement and
        update the posteri estimate and posteri error estimate.

        Args:
        measurement (float): The latest noisy measurement (=y_k).
        """

        if u_input is None:
            u_input = np.zeros((y_measurement.shape[1], 1))

        ## Time Update (Prediction)
        # x_hat_predicted = A * x_hat_posteri + B * u
        priori_estimate = \
            np.dot(
                self.a_matrix,
                self.posteri_estimate
            ) \
            + np.dot(
                self.b_matrix,
                u_input
            )
        # P_predicted = A * P_posteri * A.T + Q
        priori_error_estimate = \
            np.dot(
                self.a_matrix,
                np.dot(
                    self.posteri_error_estimate,
                    self.a_matrix.T
                )
            ) \
            + self.process_variance

        ## Measurement Update (Correction)
        # K = P_predicted * H.T * (H * P_predicted * H.T + R).inverse()
        blending_factor = \
            np.dot(
                np.dot(
                    priori_error_estimate,
                    self.c_matrix.T
                ),
                np.linalg.inv(
                    np.dot(
                        np.dot(
                            self.c_matrix,
                            priori_error_estimate
                        ),
                        self.c_matrix.T
                    ) \
                    + self.estimated_measurement_variance
                )
            )

        # x_hat_posteri = x_hat_predicted + K * (z - H * x_hat_predicted)
        self.posteri_estimate = \
            priori_estimate \
            + np.dot(
                blending_factor,
                (
                    y_measurement \
                    - np.dot(
                        self.c_matrix,
                        priori_estimate
                    )
                )
            )

        # P_posteri = (I - K * H) * P_predicted
        self.posteri_error_estimate = \
            np.dot(
                (
                    np.eye(self.c_matrix.shape[1]) \
                    - np.dot(
                        blending_factor,
                        self.c_matrix
                    )
                ),
                priori_error_estimate)


    def get_latest_estimated_measurement(self) -> np.ndarray:
        """
        Get the latest estimated measurement.

        Returns:
        float: The latest estimated measurement.
        """
        return self.posteri_estimate


    def filter_data(
        self,
        measurements_data: pd.Series,
        input_data: pd.DataFrame
        ) -> pd.Series:
        """Apply the kalman filter to the strain data.

        Args:
        measurements_data (pd.Series): The strain data to apply the kalman filter to.

        Returns:
        pd.Series: The filtered strain data.
        """
        if all(input_data.index != measurements_data.index):
            print('The measurements and input data must have the same index. Temperature set to 0')
            input_data = \
                pd.DataFrame(
                    {'input':np.zeros(1)},
                    index=measurements_data.index
                )

        filtered_data = pd.Series()
        for index, row in measurements_data.iteritems():
            self.input_latest_noisy_measurement(
                row,
                input_data.loc[index])
            filtered_data.loc[index] = \
                self.get_latest_estimated_measurement()

        self.filtered_data = filtered_data
        return filtered_data
    
    def apply_filter(
        self,
        measurements_data: pd.DataFrame,
        input_data: pd.DataFrame
        ) -> pd.DataFrame:
        """Apply the kalman filter for
        temperature compensation of the strain data.

        Args:
        measurements_data (pd.Series): The strain data to apply the kalman filter to.

        Returns:
        pd.Series: The filtered strain data.
        """
        filtered_data = self.filter_data(measurements_data, input_data)
        compensated_data = measurements_data - filtered_data
        return compensated_data

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
