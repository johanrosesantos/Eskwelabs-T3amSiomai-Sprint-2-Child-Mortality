import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import imblearn

# Define the Streamlit app
def main():
    st.title("Child Mortality Risk Prediction")

    # Determine the directory of the current script
    current_dir = os.path.dirname(__file__)

    # Construct the path to the preprocessed_dhs_dummies.csv file
    mortality_csv_path = os.path.join(current_dir, "data", "mortality_data.csv")

    # Load the mortality dataset from CSV
    try:
        mortality_df = pd.read_csv(mortality_csv_path)
    except FileNotFoundError:
        st.error(f"The mortality_data.csv file was not found at {mortality_csv_path}. Please make sure it is in the correct directory.")
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
    st.write("Here is the DHS Program child mortality dataset used for the prediction:")
    st.write(mortality_df)

    # Extract feature columns from the dataframe
    feature_columns = ["rural",
                       "region_mindanao",
                       "region_visayas",
                       "householdhead_female",
                       "wealth_poorer",
                       "wealth_middle",
                       "wealth_richer",
                       "wealth_richest",
                       "freqtv_lessthanonce",
                       "freqtv_atleasonce",
                       "freqradio_lessthanonce",
                       "freqradio_atleasonce",
                       "toilet_unimproved",
                       "toilet_open defecation",
                       "toilet_unknown",
                       "drinkingwater_unimproved",
                       "drinkingwater_unknown",
                       "motherage_20-24",
                       "motherage_25-29",
                       "motherage_30-34",
                       "motherage_35-39",
                       "motherage_40-44",
                       "motherage_45-49",
                       "mothereduc_primary",
                       "mothereduc_secondary",
                       "mothereduc_higher",
                       "mother_working",
                       "total_children_born",
                       "age_first_birth",
                       "total_births_last5years",
                       "contraceptive_folk",
                       "contraceptive_traditional",
                       "contraceptive_modern",
                       "breastfeeding_never",
                       "breastfeeding_still",
                       "child_sex_female",
                       "child_age_months",
                       "twin_1st",
                       "twin_2nd",
                       "preceeding_birthinterval_months",
                       "childsize_larger",
                       "childsize_average",
                       "childsize_smaller",
                       "childsize_verysmall",
                       "childsize_unknown"]
    mortality_data = mortality_df[feature_columns].values

    # User input for new data
    st.header("Check your child's risk to child mortality")
    
    ## Household Information
    st.write("")
    st.subheader("Household Information")
    
    rural_mapping = {"Urban Area": 0, "Rural Area": 1}
    selected_rural = st.selectbox("Where are you located?", ["Urban Area", "Rural Area"])
    rural = rural_mapping[selected_rural]
    
    ### Contraceptive
    contra_mapping = {
        "Folkloric method": "contraceptive_folk",
        "Traditional method": "contraceptive_traditional",
        "Modern method": "contraceptive_modern"}
    # Display select box for selection
    selected_contra = st.selectbox("What kind of contraceptive method do you or your partner use?", ["None", "Folkloric method", "Traditional method", "Modern method"], index=0)
    # Assign values based on selection
    for contra, element in contra_mapping.items():
        if selected_contra == contra:
            globals()[element] = 1
        else:
            globals()[element] = 0
    st.write("*Folkloric method - includes abdominal massage, amulet, bato-balani, asugi, mixtures, laxatives, salt, herbs and spiritual/cultural practices*")
    st.write("*Traditional method - includes abstinence, rhythmic or calendar method and withdrawal*")
    st.write("*Modern method - includes pill, IUD, injection, implants, female/male sterilization, male/female condom, LAM, and emergency contraception*")
    
    ### Breastfeeding
    breastfeed_mapping = {
        "Never": "breastfeeding_never",
        "Yes, up to now": "breastfeeding_still"}
    # Display select box for selection
    selected_breastfeed = st.selectbox("Have you ever breastfed your child?", ["Never", "Yes, before", "Yes, up to now"], index=0)
    # Assign values based on selection
    for breastfeed, element in breastfeed_mapping.items():
        if selected_breastfeed == breastfeed:
            globals()[element] = 1
        else:
            globals()[element] = 0
            
    ## Child Information
    st.write("")
    st.subheader("Child Information")

    sex_mapping = {"Male": 0, "Female": 1}
    selected_sex = st.selectbox("Select the child's sex", ["Male", "Female"])
    child_sex_female = sex_mapping[selected_sex]
    
    #child_age_months =
    #twin_1st =
    #twin_2nd =
    #preceeding_birthinterval_months =

    ### Child size
    childsize_mapping = {
        "Larger": "childsize_larger",
        "Average": "childsize_average",
        "Smaller": "childsize_smaller",
        "Very Small": "childsize_verysmall",
        "Unknown": "childsize_unknown"}
    # Display select box for child size selection
    selected_childsize = st.selectbox("Select the child's size", ["Larger", "Average", "Smaller", "Very Small", "Unknown"], index=4)
    # Assign values based on selected child size
    for size, element in childsize_mapping.items():
        if selected_childsize == size:
            globals()[element] = 1
        else:
            globals()[element] = 0

    # Prepare new input for prediction
    new_data = np.array([["rural",
                         "region_mindanao",
                         "region_visayas",
                         "householdhead_female",
                         "wealth_poorer",
                         "wealth_middle",
                         "wealth_richer",
                         "wealth_richest",
                         "freqtv_lessthanonce",
                         "freqtv_atleasonce",
                         "freqradio_lessthanonce",
                         "freqradio_atleasonce",
                         "toilet_unimproved",
                         "toilet_open defecation",
                         "toilet_unknown",
                         "drinkingwater_unimproved",
                         "drinkingwater_unknown",
                         "motherage_20-24",
                         "motherage_25-29",
                         "motherage_30-34",
                         "motherage_35-39",
                         "motherage_40-44",
                         "motherage_45-49",
                         "mothereduc_primary",
                         "mothereduc_secondary",
                         "mothereduc_higher",
                         "mother_working",
                         "total_children_born",
                         "age_first_birth",
                         "total_births_last5years",
                         "contraceptive_folk",
                         "contraceptive_traditional",
                         "contraceptive_modern",
                         "breastfeeding_never",
                         "breastfeeding_still",
                         "child_sex_female",
                         "child_age_months",
                         "twin_1st",
                         "twin_2nd",
                         "preceeding_birthinterval_months",
                         "childsize_larger",
                         "childsize_average",
                         "childsize_smaller",
                         "childsize_verysmall",
                         "childsize_unknown"]])

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
