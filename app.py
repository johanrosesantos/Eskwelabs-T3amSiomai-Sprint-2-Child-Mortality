import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris

# Define the Streamlit app
def main():
    st.title("Iris Species Prediction")

    # Load the Iris dataset and model
    with open("iris_data.pkl", "rb") as f:
        iris = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Display dataset
    st.write("Here is the Iris dataset used for predictions:")
    st.write(iris.data)

    # User input for new data
    st.header("Enter New Iris Flower Data")
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Prepare new input for prediction
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Batch process - combine dataset with new input
    full_data = np.vstack([iris.data, new_data])

    # Predict the species
    if st.button("Predict Species"):
        predictions = model.predict(full_data)
        new_prediction = predictions[-1]
        species = iris.target_names[new_prediction]
        st.write(f"The predicted species for the new input is: {species}")

if __name__ == "__main__":
    main()
