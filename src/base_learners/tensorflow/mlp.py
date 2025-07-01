import tensorflow as tf


class MLP:
    def __init__(self, input_shape, output_shape):
        """
        Initialize the MLP model.
        Args:
            input_shape (int): The number of features in the input data.
            output_shape (int): The number of output classes or regression targets.
        """
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    32, activation="relu", input_shape=(input_shape,)
                ),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(output_shape, activation="sigmoid"),
            ]
        )

        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def fit(
        self,
        X,
        y,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        sample_weight=None,
    ):
        """
        Fit the MLP model to the training data.
        Args:
            X: Feature data.
            y: Target labels.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Size of the batches used in training.
            validation_split (float): Fraction of the training data to be used as validation data.
            sample_weight: Optional weights for each sample.
        """
        self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            sample_weight=sample_weight,
        )

    def predict(self, X):
        """
        Predict using the MLP model.
        Args:
            X: Feature data.
        Returns:
            Predictions from the model.
        """
        return self.model.predict(X)

    def summary(self):
        """
        Print the summary of the model architecture.
        Returns:
            Summary of the model.
        """
        return self.model.summary()

    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.
        Args:
            X: Feature data.
            y: Target labels.
        Returns:
            Loss and accuracy of the model on the given data.
        """
        return self.model.evaluate(X, y)
