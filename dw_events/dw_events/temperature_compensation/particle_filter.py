"""
particle_filter.py is a module for applying the particle filter
as a temperature compensation method.
-------------------------------------------------------------------------
Author: Maximillian Weil
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import scipy as sp
import numpy as np
import pandas as pd


@dataclass
class ParticleFilter:
    """
    A simple Particle filter implementation for multidimensional data smoothing.

    Attributes:
    num_particles (int): The number of particles used in the filter.
    transition_model (function): A function that takes in a particle and returns a new particle.
    likelihood_function (function): A function that takes in a particle and a measurement and returns the
        likelihood of the measurement given the particle.
    initial_particles (list of np.ndarrays): The initial particles of the system.
    """
    num_particles: int
    r_measurement_noise: float = 0.1
    q_process_noise: np.ndarray = np.array([0.1, 0.1])
    scale: float = 1

    def __post_init__(self):
        self.particles = np.zeros((self.num_particles, 2))
        self.weights = np.ones(self.num_particles) / self.num_particles

    def create_gaussian_particles(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        ) -> None:
        self.particles = np.empty((self.num_particles, 2))
        self.particles[:,0] = \
            mean[0] + (np.random.randn(self.num_particles) * std[0])
        self.particles[:,1] = \
            mean[1] + (np.random.randn(self.num_particles) * std[1])
    
    def predict(
            self,
            u_input: np.ndarray
            ) -> None:
        """ move according to control input u (measured temperature)
        with input noise q (std measured temperature)"""
        # update Ta
        self.particles[:, 0] += \
            u_input[1] \
            + (np.random.randn(self.num_particles) * self.q_process_noise[0])
        # update delta Ta
        self.particles[:, 1] += \
            (np.random.randn(self.num_particles) * self.q_process_noise[1])
    
    def update(
        self,
        y_measurement
        ) -> None:
        """ incorporate measurement y (measured temperature)
        with measurement noise r (std measured temperature)"""
        # compute likelihood of measurement
        distance = self.particles[:, 0] - y_measurement
        self.weights *= \
            sp.stats.expon(
                distance,
                self.r_measurement_noise
            ).pdf(self.particles[:, 0])

        self.weights += 1.e-300      # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize

    def estimate(
        self
        ):
        """returns mean and variance of the weighted particles"""
        pos = self.particles[:, 0]
        mean = np.average(pos, weights=self.weights, axis=0)
        var  = np.average((pos - mean)**2, weights=self.weights, axis=0)
        return mean, var
    
    def simple_resample(self):
        """resample particles with replacement according to weights"""
        cumulative_sum = \
            np.cumsum(self.weights)
        # normalize the cumulative sum to be in [0, 1]
        cumulative_sum /= cumulative_sum[self.num_particles-1]
        randoms = np.random.rand(self.num_particles)
        indexes = \
            np.searchsorted(cumulative_sum, randoms)
        self.particles[:] = \
            self.particles[indexes] \
            - np.array([
                np.random.exponential(scale=self.scale/np.abs(self.particles[indexes, 0]), size=self.num_particles),
                np.random.exponential(scale=self.scale/np.abs(self.particles[indexes, 1]), size=self.num_particles)
            ]).T
        self.weights[:] = self.weights[indexes]
        self.weights /= np.sum(self.weights)


    def filter(
        self,
        measurements: np.ndarray,
        input: np.ndarray = None,
        ) -> np.ndarray:
        """Filter the data using the particle filter.
        """
        predictions = []
        mean = np.array([measurements[0], 0])
        std = np.array([0.1, 0.1])
        print(mean, std)
        self.create_gaussian_particles(mean, std)
        for i in range(len(measurements)):
            self.predict(input[i])
            #print(self.particles, self.weights)
            self.update(measurements[i])
            #print(self.particles, self.weights)
            self.simple_resample()
            #print(self.particles, self.weights)
            prediction, var = self.estimate()
            #print(self.particles, self.weights)
            predictions.append(prediction)
        return np.array(predictions)


@dataclass
class ParticleFilter_GPT:
    """
    A simple Particle filter implementation for multidimensional data smoothing.

    Attributes:
    num_particles (int): The number of particles used in the filter.
    transition_model (function): A function that takes in a particle and returns a new particle.
    likelihood_function (function): A function that takes in a particle and a measurement and returns the
        likelihood of the measurement given the particle.
    initial_particles (list of np.ndarrays): The initial particles of the system.
    """
    num_particles: int
    initial_particles: List[np.ndarray]
    process_noise: float = 0.1
    measurement_noise: float = 0.1
    alpha: float = 1
    noise_skew: str = 'positive'

    def __post_init__(self):
        self.particles = np.array(self.initial_particles)

    def likelihood_function(self, y, x, measurement_noise, alpha=1):
        z = (y - x) / measurement_noise
        if measurement_noise > 0:
            if z >= 0:
                return np.exp(-np.power(z, alpha))
            else:
                return np.exp(-np.power(-z, alpha) - alpha * np.log(-z))
        else:
            return 1 if z == 0 else 0

    def transition_model_positive(self, x, process_noise):
        return np.random.weibull(1.5) * x + process_noise

    def transition_model_negative(self, x, process_noise):
        return -np.random.weibull(1.5) * x + process_noise

    def transition_model(self, x, process_noise, noise_skew):
        if noise_skew == 'positive':
            return self.transition_model_positive(x, process_noise)
        elif noise_skew == 'negative':
            return self.transition_model_negative(x, process_noise)
        else:
            raise ValueError("noise_skew should be either 'positive' or 'negative'")

    def resample(self, weights: np.ndarray) -> np.ndarray:
        """
        Resample the particles based on their weights.

        Args:
        weights (np.ndarray): The weights of the particles.

        Returns:
        np.ndarray: The resampled particles.
        """
        indices = np.random.choice(self.num_particles, self.num_particles, p=weights)
        return self.particles[indices]

    def filter_data_quick(
        self,
        measurements: np.ndarray
    ) -> np.ndarray:
        """Filter the data using the particle filter.

        Args:
        measurements (np.ndarray): The measurements to be filtered.

        Returns:
        np.ndarray: The filtered data.
        """
        num_measurements = measurements.shape[0]
        filtered_data = np.zeros(num_measurements)

        for i in range(num_measurements):
            # Propagate the particles
            for j in range(self.num_particles):
                self.particles[j] = self.transition_model(self.particles[j], self.process_noise, self.noise_skew)

            # Compute the likelihoods
            weights = np.zeros(self.num_particles)
            for j in range(self.num_particles):
                weights[j] = \
                    self.likelihood_function(
                        self.particles[j], 
                        measurements[i], 
                        self.measurement_noise, 
                        self.alpha
                    )

            # Normalize the weights
            weights /= np.sum(weights)

            # Resample the particles
            self.particles = self.resample(weights)

            # Compute the estimate of the state
            filtered_data[i] = np.mean(self.particles)

        return filtered_data
