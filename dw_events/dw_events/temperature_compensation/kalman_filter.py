"""
kalman_filter.py is a module for applying the kalman filter
as a temperature compensation method.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class KalmanFilter:
    """
    A simple Kalman filter implementation for data smoothing.

    Attributes:
    process_variance (float): The variance of the process noise, which represents the uncertainty
        of the system's state transition model.
    estimated_measurement_variance (float): The variance of the measurement noise, which represents
        the uncertainty of the measurement model.
    posteri_estimate (float): The latest estimated measurement, initialized to 0.0.
    posteri_error_estimate (float): The latest estimated error of the measurement, initialized to 1.0.
    """
    def __init__(self, process_variance, estimated_measurement_variance):
        """
        Initialize the Kalman filter with the process variance and estimated measurement variance.

        Args:
        process_variance (float): The variance of the process noise.
        estimated_measurement_variance (float): The variance of the measurement noise.
        """
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def input_latest_noisy_measurement(self, measurement):
        """
        Input the latest noisy measurement and update the posteri estimate and posteri error estimate.

        Args:
        measurement (float): The latest noisy measurement.
        """
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (priori_error_estimate + self.estimated_measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

    def get_latest_estimated_measurement(self):
        """
        Get the latest estimated measurement.

        Returns:
        float: The latest estimated measurement.
        """
        return self.posteri_estimate

        
@dataclass
class KalmanTempComp:
    """A class to apply the Kalman filter as
    a temperature compensation method for strain data.
    """
    measurement_uncertainty: float = 1e-3 # R
    delta: float = 1e-4
    dimension = 1


    def __post_init__(self):
        self.compensated_data: pd.DataFrame = field(init=False)
        self.noise_covariance\
            = self.delta / (1 - self.delta) * np.eye(self.dimension) # Q
        self.state = np.zeros((self.dimension, 1)) # x
        self.uncertainty = np.zeros((self.dimension, self.dimension)) # P
    
    def iterate(self, measurement):
        observation_matrix = np.array([measurement])[None]

        # Time Update (Prediction)
        state_hat = self.state[:, -1][..., None]
        uncertainty_hat = self.uncertainty + self.noise_covariance

        # Measurement Update (Correction)
        prediction_uncertainty = uncertainty_hat.dot(observation_matrix.T)
        kalman_gain_factor = \
            prediction_uncertainty\
            / (
                observation_matrix.dot(prediction_uncertainty)\
                + self.measurement_uncertainty
            ) # Kn
        measurement_hat = observation_matrix.dot(state_hat)
        state = state_hat + kalman_gain_factor * (measurement - measurement_hat)
        self.uncertainty = \
            (
                np.eye(self.dimension) \
                - kalman_gain_factor.dot(observation_matrix)
            ).dot(uncertainty_hat)
        self.state = np.concatenate((self.state, state), axis=1)

        return state, self.uncertainty, kalman_gain_factor, measurement_hat
        
        
    

