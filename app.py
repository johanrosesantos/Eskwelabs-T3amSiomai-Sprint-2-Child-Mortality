import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Define the Streamlit app
def main():
    st.title("Iris Species Prediction")

    # Determine the directory of the current script
    current_dir = os.path.dirname(__file__)

    # Construct the path to the iris.csv file
    iris_csv_path = os.path.join(current_dir, "data", "iris.csv")

    # Load the Iris dataset from CSV
    try:
        iris_df = pd.read_csv(iris_csv_path)
    except FileNotFoundError:
        st.error(f"The iris.csv file was not found at {iris_csv_path}. Please make sure it is in the correct directory.")
        return

    # Construct the path to the model.pkl file
    model_path = os.path.join(current_dir, "model.pkl")

    # Load the model from pickle file
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"The model.pkl file was not found at {model_path}. Please make sure it is in the correct directory.")
        return

    # Display dataset
    st.write("Here is the Iris dataset used for predictions:")
    st.write(iris_df)

    # Extract feature columns from the dataframe
    feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    iris_data = iris_df[feature_columns].values

    # User input for new data
    st.header("Enter New Iris Flower Data")
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Prepare new input for prediction
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Batch process - combine dataset with new input
    full_data = np.vstack([iris_data, new_data])

    # Predict the species
    if st.button("Predict"):
        predictions = model.predict(full_data)
        new_prediction = predictions[-1]
        species = iris_df["species"].unique()[new_prediction]
        st.write(f"The predicted species for the new input is: {species}")

if __name__ == "__main__":
    main()
