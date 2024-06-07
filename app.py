import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris

# Load the custom CSS with the background image
custom_css = """
<style>
body {
    background-image: url('https://media.istockphoto.com/id/1280979263/photo/mother-son-and-grandmother-looking-out-from-window-of-bajau-stilt-shanty-house.jpg?s=1024x1024&w=is&k=20&c=jd4zt6rxuZpfgpky3Dp3BuNz9GNujoov1DCJ3mqOWpU=');
    background-size: cover;
}
</style>
"""

# Display the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Load the iris dataset to get target names
iris = load_iris()

# Load the trained model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
def main():
    st.title("Iris Species Prediction")

    # Single user input
    # disadvantage - takes time
    
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Batch process - use holdout from dataset include user input then populate
    
    # Predict the species
    if st.button("Predict"):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        species = iris.target_names[prediction][0]
        st.write(f"The predicted species is: {species}")

if __name__ == "__main__":
    main()
