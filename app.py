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
                       "toilet_open_defecation",
                       "toilet_unknown",
                       "drinkingwater_unimproved",
                       "drinkingwater_unknown",
                       "mother_age_20_24",
                       "mother_age_25_29",
                       "mother_age_30_34",
                       "mother_age_35_39",
                       "mother_age_40_44",
                       "mother_age_45_49",
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
                       "preceding_birthinterval_months",
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
    selected_rural = st.selectbox("Where are you located?", ["Urban Area", "Rural Area"], index=0)
    rural = rural_mapping[selected_rural]

    ### Region
    region_mapping = {
        "Visayas": "region_visayas",
        "Mindanao": "region_mindanao",
        "Modern method": "contraceptive_modern"}
    # Display select box for selection
    selected_region = st.selectbox("What region are you residing in?", ["Luzon", "Visayas", "Mindanao"], index=0)
    # Assign values based on selection
    for region, element in region_mapping.items():
        if selected_region == region:
            globals()[element] = 1
        else:
            globals()[element] = 0

    ### Household Head
    head_mapping = {"Male": 0, "Female": 1}
    selected_head = st.selectbox("What is the sex of the household head?", ["Male", "Female"], index=0)
    householdhead_female = head_mapping[selected_head]

    ### Wealth index
    wealth_mapping = {
        "Php10,957-Php43,828": "wealth_poorer",
        "Php43,828-Php76,669": "wealth_middle",
        "Php76,669-Php219,140": "wealth_richer",
        "more than Php219,140": "wealth_richest"}
    # Display select box for selection
    selected_wealth = st.selectbox("What is your household's total income?", ["less than Php10,957", "Php10,957-Php43,828", "Php43,828-Php76,669", "Php76,669-Php219,140","more than Php219,140"], index=2)
    # Assign values based on selection
    for wealth, element in wealth_mapping.items():
        if selected_wealth == wealth:
            globals()[element] = 1
        else:
            globals()[element] = 0

    ### Frequency TV
    tv_mapping = {
        "Less than once a week": "freqtv_lessthanonce",
        "At least once a week": "freqtv_atleasonce"}
    # Display select box for selection
    selected_tv = st.selectbox("How frequent do you watch TV?", ["I don't watch TV", "Less than once a week", "At least once a week"], index=2)
    # Assign values based on selection
    for tv, element in tv_mapping.items():
        if selected_tv == tv:
            globals()[element] = 1
        else:
            globals()[element] = 0
    
    ### Frequency Radio
    rad_mapping = {
        "Less than once a week": "freqradio_lessthanonce",
        "At least once a week": "freqradio_atleasonce"}
    # Display select box for selection
    selected_rad = st.selectbox("How frequent do you listen to radio?", ["I don't listen to radio", "Less than once a week", "At least once a week"], index=2)
    # Assign values based on selection
    for rad, element in rad_mapping.items():
        if selected_rad == rad:
            globals()[element] = 1
        else:
            globals()[element] = 0

    ### Toilet Facility
    toilet_mapping = {
        "Unimproved Toilet Facility": "toilet_unimproved",
        "Open Defecation": "toilet_open_defecation",
        "I don't know": "toilet_unknown"}
    # Display select box for selection
    selected_toilet = st.selectbox("What kind of toilet do you use?", ["Unimproved Toilet Facility", "Unimproved Toilet Facility", "Open Defecation", "I don't know"], index=0)
    # Assign values based on selection
    for toilet, element in toilet_mapping.items():
        if selected_toilet == toilet:
            globals()[element] = 1
        else:
            globals()[element] = 0
    st.write("*Improved Toilet Facility - flush/pour flush toilet connected to piped sewer/septic tank/pit latrine, pit latrine with slab, composting toilet*")
    st.write("*Unimproved Toilet Facility - flush/pour flush toilet NOT connected to piped sewer/septic tank/pit latrine, open pit, bucket, hanging toilet/latrine*")
    st.write("*Open Defecation - no facility/toilet, bush, field*")

    ### Drinking Water
    water_mapping = {
        "Unimproved Drinking Water Source": "drinkingwater_unimproved",
        "I don't know": "drinkingwater_unknown"}
    # Display select box for selection
    selected_water = st.selectbox("Where do you get your drinking water?", ["Improved Drinking Water Source", "Unimproved Drinking Water Source", "I don't know"], index=0)
    # Assign values based on selection
    for water, element in water_mapping.items():
        if selected_water == water:
            globals()[element] = 1
        else:
            globals()[element] = 0
    st.markdown("*- Improved Drinking Water Source - piped, tube well, borehole, protected dug well/spring, rainwater, tanker truck, bottled water, water refilling station*")
    st.markdown("*- Unimproved Drinking Water Source - unprotected dug well/spring, surface water*")



    
    ## Mother Information
    st.write("")
    st.subheader("Mother's Information")

    ### Mother's age
    mage_mapping = {
        "20-24": "mother_age_20_24",
        "25-29": "mother_age_25_29",
        "30-34": "mother_age_30_34",
        "35-39": "mother_age_35_39",
        "40-44": "mother_age_40_44",
        "45-49": "mother_age_45_49"}
    # Display select box for selection
    selected_mage = st.selectbox("What is the mother's age in years?", ["less than 20","20-24", "25-29", "30-34", "35-39","40-44","45-49","more than 49"], index=1)
    # Assign values based on selection
    for mage, element in mage_mapping.items():
        if selected_mage == mage:
            globals()[element] = 1
        else:
            globals()[element] = 0

    ### Mother's Education
    educ_mapping = {
        "Elementary": "mothereduc_primary",
        "Highschool": "mothereduc_secondary",
        "College or Higher": "mothereduc_higher"}
    # Display select box for selection
    educ_mage = st.selectbox("What is the mother's highest educational attainment?", ["None", "Elementary", "Highschool", "College or Higher"], index=1)
    # Assign values based on selection
    for educ, element in educ_mapping.items():
        if selected_educ == educ:
            globals()[element] = 1
        else:
            globals()[element] = 0
        
    # Working
    work_mapping = {"No": 0, "Yes": 1}
    selected_work = st.selectbox("Are you currently employed?", ["No", "Yes"], index=0)
    mother_working = workl_mapping[selected_work]

    # Total Children Born
    total_children_born = st.number_input("How many children have you given birth to?", min_value=1, max_value=50, value=1, step=1)

    # Age at first birth
    age_first_birth = st.number_input("At what age did you have your first child?", min_value=1, max_value=50, value=1, step=1)

    # Total births in 5 years
    total_births_last5years = st.number_input("How many times have you given birth in the last 5 years?", min_value=1, max_value=10, value=1, step=1)

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





    ## Child's Information
    st.write("")
    st.subheader("Child's Information")

    # Child sex
    sex_mapping = {"Male": 0, "Female": 1}
    selected_sex = st.selectbox("Select the child's sex", ["Male", "Female"])
    child_sex_female = sex_mapping[selected_sex]

    # Child age in months
    child_age_months = st.number_input("How old is your child in months?", min_value=0, max_value=60, value=1, step=1)

    # Twin Birth
    twin_mapping = {
        "Yes, 1st born": "twin_1st",
        "Yes, last born": "twin_2nd"}
    # Display select box for child size selection
    selected_twin = st.selectbox("Is this child a twin, or part of a multiple birth?", ["No", "Yes, 1st born", "Yes, last born"], index=0)
    # Assign values based on selected child size
    for twin, element in twin_mapping.items():
        if selected_twin == twin:
            globals()[element] = 1
        else:
            globals()[element] = 0
    
    # Preceding birth
    preceeding_birthinterval_months = st.number_input("How many months apart are your current child and your previous child?", min_value=0, max_value=500, value=12, step=1)

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
    new_data = np.array([[rural,
                          region_mindanao,
                          region_visayas,
                          householdhead_female,
                          wealth_poorer,
                          wealth_middle,
                          wealth_richer,
                          wealth_richest,
                          freqtv_lessthanonce,
                          freqtv_atleasonce,
                          freqradio_lessthanonce,
                          freqradio_atleasonce,
                          toilet_unimproved,
                          toilet_open_defecation,
                          toilet_unknown,
                          drinkingwater_unimproved,
                          drinkingwater_unknown,
                          mother_age_20_24,
                          mother_age_25_29,
                          mother_age_30_34,
                          mother_age_35_39,
                          mother_age_40_44,
                          mother_age_45_49,
                          mothereduc_primary,
                          mothereduc_secondary,
                          mothereduc_higher,
                          mother_working,
                          total_children_born,
                          age_first_birth,
                          total_births_last5years,
                          contraceptive_folk,
                          contraceptive_traditional,
                          contraceptive_modern,
                          breastfeeding_never,
                          breastfeeding_still,
                          child_sex_female,
                          child_age_months,
                          twin_1st,
                          twin_2nd,
                          preceeding_birthinterval_months,
                          childsize_larger,
                          childsize_average,
                          childsize_smaller,
                          childsize_verysmall,
                          childsize_unknown]])

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
