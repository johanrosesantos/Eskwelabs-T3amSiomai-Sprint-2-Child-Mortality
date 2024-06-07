import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Ensure the required library is installed
try:
    import imblearn
except ImportError:
    st.error("The 'imblearn' library is required but not installed. Please install it using 'pip install imbalanced-learn' and try again.")
    raise

# Define the Streamlit app
def main():
    st.title("Child Mortality Risk Prediction")

    # Determine the directory of the current script
    current_dir = os.path.dirname(__file__)

    # Construct the path to the preprocessed_dhs_dummies.csv file
    mortality_csv_path = os.path.join(current_dir, "data", "preprocessed_dhs_dummies.csv")

    # Load the mortality dataset from CSV
    try:
        mortality_df = pd.read_csv(mortality_csv_path)
    except FileNotFoundError:
        st.error(f"The preprocessed_dhs_dummies.csv file was not found at {mortality_csv_path}. Please make sure it is in the correct directory.")
        return

    # Construct the path to the adaboost.pkl file
    model_path = os.path.join(current_dir, "adaboost.pkl")

    # Load the model from pickle file
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"The adaboost.pkl file was not found at {model_path}. Please make sure it is in the correct directory.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return

    # Display dataset
    st.write("Here is the child mortality dataset used for the prediction:")
    st.write(mortality_df)

    # Extract feature columns from the dataframe
    # Replace with actual feature columns from your dataset
    feature_columns = ["feature1", "feature2", "feature3", "feature4"]
    mortality_data = mortality_df[feature_columns].values

    # User input for new data
    st.header("Enter New Data for Prediction")
    feature1 = st.number_input("Feature 1", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    feature2 = st.number_input("Feature 2", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    feature3 = st.number_input("Feature 3", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    feature4 = st.number_input("Feature 4", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

    # Prepare new input for prediction
    new_data = np.array([[feature1, feature2, feature3, feature4]])

    # Batch process - combine dataset with new input
    full_data = np.vstack([mortality_data, new_data])

    # Predict the outcome
    if st.button("Predict"):
        try:
            predictions = model.predict(full_data)
            new_prediction = predictions[-1]
            st.write(f"The predicted outcome for the new input is: {new_prediction}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
