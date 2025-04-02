import requests
import io
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.base_learners.tensorflow import MLP

# ------------------------------------------------------------------
# 1. Fetch a dummy dataset from the web
# ------------------------------------------------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
response = requests.get(url)
response.raise_for_status()  # Ensure the download was successful

# The file has no headers, so we’ll manually assign columns
column_names = [
    "num_pregnant",
    "plasma_glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
    "outcome"  # 1 = positive test for diabetes, 0 = negative
]

# Load CSV data into a pandas DataFrame
df = pd.read_csv(io.StringIO(response.text), header=None, names=column_names)

# ------------------------------------------------------------------
# 2. Preprocess: split & scale
# ------------------------------------------------------------------
# Separate features (X) and label (y)
X = df.drop("outcome", axis=1).values
y = df["outcome"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (optional but often helps neural nets converge)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------------------------------------------
# 3. Build a simple Keras model
# ------------------------------------------------------------------
model = MLP(input_shape=X_train.shape[1], output_shape=1)

model.summary()  # Print a simple summary

# ------------------------------------------------------------------
# 4. Train the model
# ------------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=10,            # small number of epochs for demo
    batch_size=32, 
    validation_split=0.1  # track validation performance
)

# ------------------------------------------------------------------
# 5. Evaluate on the test set
# ------------------------------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# ------------------------------------------------------------------
# 6. Optional: Check GPU usage (Metal)
# ------------------------------------------------------------------
print("Available physical devices:", tf.config.list_physical_devices())

# If a GPU is visible, it should say something like:
#   [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
#    PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
