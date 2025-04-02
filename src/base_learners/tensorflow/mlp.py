import tensorflow as tf

class MLP:
    def __init__(self, input_shape, output_shape):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(output_shape, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def fit(self, X, y, epochs=100, batch_size=10, validation_split=0.1):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def predict(self, X):
        return self.model.predict(X)
    
    def summary(self):
        return self.model.summary()
    
    def evaluate(self, X, y):
        return self.model.evaluate(X, y)