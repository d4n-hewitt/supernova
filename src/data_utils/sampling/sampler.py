from random import choices, sample


class Sampler:

    def __init__(self, data):
        """
        Initialize the Sampler with data.

        Args:
            data (list): The data to sample from.
        """
        self.data = data

    def calculate_sample_size(self, sample_fraction):
        """
        Calculate the sample size based on the fraction of the data.

        Args:
            data (list): The data to sample from.
            sample_fraction (float): The fraction of the data to sample.

        Returns:
            int: The calculated sample size.
        """
        if not (0 < sample_fraction <= 1):
            raise ValueError("sample_fraction must be between 0 and 1.")
        return int(len(self.data) * sample_fraction)

    def sample_with_replacement(self, num_samples):
        """
        Sample data with replacement.

        Args:
            data (list): The data to sample from.
            num_samples (int): The number of samples to draw.

        Returns:
            list: A list of sampled data.
        """
        if num_samples > len(self.data):
            raise ValueError("num_samples must be less than length of data.")
        return choices(self.data, k=num_samples)

    def sample_without_replacement(self, num_samples):
        """
        Sample data without replacement.

        Args:
            data (list): The data to sample from.
            num_samples (int): The number of samples to draw.

        Returns:
            list: A list of sampled data.
        """
        if num_samples > len(self.data):
            raise ValueError("num_samples must be less than length of data.")
        return sample(self.data, k=num_samples)


# Example usage
