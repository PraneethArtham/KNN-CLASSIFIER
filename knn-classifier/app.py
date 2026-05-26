import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Create folders automatically

os.makedirs("data", exist_ok=True)

os.makedirs("models", exist_ok=True)
# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("KNN Classifier Application")

st.write("Iris Flower Prediction using KNN Classifier")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["target"] = iris.target

# ---------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------

df.to_csv("data/iris.csv", index=False)

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------

X = df.drop("target", axis=1)

y = df["target"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Save scaler

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

# Save model

pickle.dump(
    model,
    open("models/knn_model.pkl", "wb")
)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

st.subheader("Model Accuracy")

st.write(f"Accuracy : {accuracy:.2f}")

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

st.subheader("Enter Flower Details")

sepal_length = st.slider(
    "Sepal Length",
    4.0,
    8.0,
    5.0
)

sepal_width = st.slider(
    "Sepal Width",
    2.0,
    5.0,
    3.0
)

petal_length = st.slider(
    "Petal Length",
    1.0,
    7.0,
    4.0
)

petal_width = st.slider(
    "Petal Width",
    0.1,
    3.0,
    1.0
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if st.button("Predict Flower"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Scaling input

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    flower_names = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    st.success(
        f"Predicted Flower : {flower_names[prediction[0]]}"
    )
