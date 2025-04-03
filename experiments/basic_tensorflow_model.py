import io

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.base_learners.tensorflow import MLP
from src.ensembling.bagging import BaggingClassifier

# ------------------------------------------------------------------
# 1. Fetch a dummy dataset from the web
# ------------------------------------------------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"  # noqa
response = requests.get(url)
response.raise_for_status()

column_names = [
    "num_pregnant",
    "plasma_glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "outcome",
]

df = pd.read_csv(io.StringIO(response.text), header=None, names=column_names)

# ------------------------------------------------------------------
# 2. Preprocess: split & scale
# ------------------------------------------------------------------
X = df.drop("outcome", axis=1).values
y = df["outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------------
# 3. Build a simple Keras model
# ------------------------------------------------------------------
model = MLP(input_shape=X_train.shape[1], output_shape=1)

model.summary()

print("Creating ensemble")
ensemble = BaggingClassifier(
    base_estimator=MLP,
    n_estimators=2,
    sample_fraction=0.7,
)

# ------------------------------------------------------------------
# 4. Train the model
# ------------------------------------------------------------------
# history = model.fit(
#     X_train,
#     y_train,
#     epochs=10,  # small number of epochs for demo
#     batch_size=32,
#     validation_split=0.1,  # track validation performance
# )

print("Training ensemble")
ensemble.fit(X_train, y_train)

# ------------------------------------------------------------------
# 5. Evaluate on the test set
# ------------------------------------------------------------------
loss, accuracy = ensemble.evaluate(X_test, y_test)

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
