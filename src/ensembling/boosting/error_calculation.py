import numpy as np


class DBErrorCalculator:
    """
    Class for calculating errors and sample weights in boosting algorithms.
    This class supports different error types and sample schemes for flexibility in boosting implementations.
    """

    def __init__(
        self,
        error_type="absolute",
        sample_scheme="linear",
        error_combination="uniform",
    ):
        self.error_type = error_type
        self.sample_scheme = sample_scheme
        self.error_combination = error_combination
        self.previous_errors = []

        self.error_type_dict = {
            "absolute": lambda x: np.abs(x),
            "squared": lambda x: np.square(x),
        }

        self.sample_scheme_dict = {
            "linear": lambda x: np.abs(x),
            "exponential": lambda x: np.exp(x),
        }

        self.error_combination_dict = {
            "uniform": self._uniform_combination,
            "linear_decay": self._linear_decay_combination,
        }

    def calc_error(self, predictions, y):
        """
        Calculate the error between predictions and true labels.

        Args:
            predictions (np.ndarray): Predicted values
            y (np.ndarray): Ground truth labels

        Returns:
            np.ndarray: Computed error
        """
        differences = predictions - y

        if self.error_type not in self.error_type_dict:
            raise ValueError(
                f"Unsupported error type: {self.error_type}. "
                f"Supported types are: {list(self.error_type_dict.keys())}"
            )

        errors = self.error_type_dict[self.error_type](differences)
        self.previous_errors.append(errors)

    def _uniform_combination(self, errors):
        """
        Combine errors using a uniform method.

        Args:
            errors (list of np.ndarray): List of error arrays

        Returns:
            np.ndarray: Combined error values
        """
        return np.mean(errors, axis=0)

    def _linear_decay_combination(self, errors):
        """
        Combine errors using a linear decay method.
        Args:
            errors (list of np.ndarray): List of error arrays
        Returns:
            np.ndarray: Combined error values
        """
        weights = np.arange(1, errors.shape[0] + 1)
        weights = weights / weights.sum()
        return np.average(errors, axis=0, weights=weights)

    def _combine_errors(self):
        """
        Combine previous errors using the specified error combination method.

        Returns:
            np.ndarray: Combined error values
        """
        if not self.previous_errors:
            raise ValueError(
                "No previous errors to combine. Ensure calc_error has been called first."
            )
        errors = np.stack(self.previous_errors)

        if self.error_combination not in self.error_combination_dict:
            raise ValueError(
                f"Unsupported error combination method: {self.error_combination}. "
                f"Supported methods are: {list(self.error_combination_dict.keys())}"
            )

        return self.error_combination_dict[self.error_combination](errors)

    def compute_sample_weights(self):
        """
        Compute sample weights from errors.

        Returns:
            np.ndarray: Normalised sample weights
        """
        combined_errors = self._combine_errors()
        if self.sample_scheme not in self.sample_scheme_dict:
            raise ValueError(
                f"Unsupported sample scheme: {self.sample_scheme}. "
                f"Supported schemes are: {list(self.sample_scheme_dict.keys())}"
            )

        weights = (
            self.sample_scheme_dict[self.sample_scheme](combined_errors)
            + 1e-10
        )
        return weights / np.sum(weights)
