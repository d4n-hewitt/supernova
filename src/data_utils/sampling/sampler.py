import random

import numpy as np


class Sampler:
    def __init__(self, X, y):
        """
        Initialize the Sampler with X and y data.

        Args:
            X (list or numpy array): The feature data to sample from.
            y (list or numpy array): The target data to sample from.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        self.X = X
        self.y = y
        self.n = len(X)

    def calculate_sample_size(self, sample_fraction):
        """
        Calculate the sample size based on the fraction of the data.

        Args:
            sample_fraction (float): The fraction of the data to sample.

        Returns:
            int: The calculated sample size.
        """
        if not (0 < sample_fraction <= 1):
            raise ValueError("sample_fraction must be between 0 and 1.")
        return int(self.n * sample_fraction)

    def sample_indices_with_replacement(self, num_samples):
        """
        Sample indices with replacement.

        Args:
            num_samples (int): The number of samples to draw.

        Returns:
            list of int: A list of sampled indices.
        """
        if num_samples > self.n:
            raise ValueError("num_samples must be <= length of data.")

        return random.choices(range(self.n), k=num_samples)

    def sample_indices_without_replacement(self, num_samples):
        """
        Sample indices without replacement.

        Args:
            num_samples (int): The number of samples to draw.

        Returns:
            list of int: A list of sampled indices.
        """
        if num_samples > self.n:
            raise ValueError("num_samples must be <= length of data.")

        return random.sample(range(self.n), k=num_samples)

    def sample_with_replacement(self, num_samples):
        """
        Sample X and y with replacement.

        Args:
            num_samples (int): The number of samples to draw.

        Returns:
            (list, list): Two lists (or arrays) containing sampled X and y.
        """
        idxs = self.sample_indices_with_replacement(num_samples)
        sampled_X = np.asarray([self.X[i] for i in idxs])
        sampled_y = np.asarray([self.y[i] for i in idxs])
        return sampled_X, sampled_y

    def sample_without_replacement(self, num_samples):
        """
        Sample X and y without replacement.

        Args:
            num_samples (int): The number of samples to draw.

        Returns:
            (list, list): Two lists (or arrays) containing sampled X and y.
        """
        idxs = self.sample_indices_without_replacement(num_samples)
        sampled_X = np.asarray([self.X[i] for i in idxs])
        sampled_y = np.asarray([self.y[i] for i in idxs])
        return sampled_X, sampled_y
