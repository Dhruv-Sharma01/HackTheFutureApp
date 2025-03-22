import streamlit as st
import numpy as np
import pandas as pd
import lightgbm as lgb

# ---------------------------------------------------
# Load the LightGBM model using its native Booster interface.
model = lgb.Booster(model_file='model.pkl')

st.title("Household Total Expense Prediction")
st.markdown("Enter your household details and person-specific details below:")

# ===================================================
# HOUSEHOLD-LEVEL INPUTS
# ===================================================

st.header("Household-Level Information")

# Example household-level fields
Sector = st.number_input("Sector", value=1, step=1)
# Map Indian state names to numeric codes (1..29) based on alphabetical order.
state_mapping = {
    "Andhra Pradesh": 1,
    "Arunachal Pradesh": 2,
    "Assam": 3,
    "Bihar": 4,
    "Chhattisgarh": 5,
    "Goa": 6,
    "Gujarat": 7,
    "Haryana": 8,
    "Himachal Pradesh": 9,
    "Jharkhand": 10,
    "Karnataka": 11,
    "Kerala": 12,
    "Madhya Pradesh": 13,
    "Maharashtra": 14,
    "Manipur": 15,
    "Meghalaya": 16,
    "Mizoram": 17,
    "Nagaland": 18,
    "Odisha": 19,
    "Punjab": 20,
    "Rajasthan": 21,
    "Sikkim": 22,
    "Tamil Nadu": 23,
    "Telangana": 24,
    "Tripura": 25,
    "Uttar Pradesh": 26,
    "Uttarakhand": 27,
    "West Bengal": 28,
    "Other": 29
}
state_name = st.selectbox("Select your State", options=list(state_mapping.keys()))
State = state_mapping[state_name]
nss_region = st.number_input("NSS-Region", value=213, step=1)
District = st.number_input("District", value=6, step=1)
Household_Type = st.number_input("Household Type", value=6, step=1)
Religion = st.number_input("Religion of the head of the household", value=1, step=1)
Social_Group = st.number_input("Social Group of the head of the household", value=1, step=1)
HH_Size = st.number_input("HH Size (For FDQ)", value=2, step=1)
NCO_3D = st.number_input("NCO_3D", value=963.0)
NIC_5D = st.number_input("NIC_5D", value=1110.0)

# Online purchase behavior (binary inputs)
st.subheader("Online Purchase Behavior (Last 365 Days)")
Is_online_Clothing = st.selectbox("Online Clothing Purchased Last365?", options=[0, 1], index=0)
Is_online_Footwear = st.selectbox("Online Footwear Purchased Last365?", options=[0, 1], index=0)
Is_online_Furniture = st.selectbox("Online Furniture fixtures Purchased Last365?", options=[0, 1], index=0)
Is_online_Mobile = st.selectbox("Online Mobile Handset Purchased Last365?", options=[0, 1], index=0)
Is_online_Personal_Goods = st.selectbox("Online Personal Goods Purchased Last365?", options=[0, 1], index=0)
Is_online_Recreation = st.selectbox("Online Recreation Goods Purchased Last365?", options=[0, 1], index=0)
Is_online_Household_Appliances = st.selectbox("Online Household Appliances Purchased Last365?", options=[0, 1], index=0)
Is_online_Crockery = st.selectbox("Online Crockery Utensils Purchased Last365?", options=[0, 1], index=0)
Is_online_Sports = st.selectbox("Online Sports Goods Purchased Last365?", options=[0, 1], index=0)
Is_online_Medical = st.selectbox("Online Medical Equipment Purchased Last365?", options=[0, 1], index=0)
Is_online_Bedding = st.selectbox("Online Bedding Purchased Last365?", options=[0, 1], index=0)

# Household assets (binary inputs)
st.subheader("Household Assets")
Is_HH_Have_Television = st.selectbox("Household has Television?", options=[0, 1], index=1)
Is_HH_Have_Radio = st.selectbox("Household has Radio?", options=[0, 1], index=0)
Is_HH_Have_Laptop_PC = st.selectbox("Household has Laptop/PC?", options=[0, 1], index=0)
Is_HH_Have_Mobile_handset = st.selectbox("Household has Mobile handset?", options=[0, 1], index=1)
Is_HH_Have_Bicycle = st.selectbox("Household has Bicycle?", options=[0, 1], index=0)
Is_HH_Have_Motorcycle_scooter = st.selectbox("Household has Motorcycle/Scooter?", options=[0, 1], index=0)
Is_HH_Have_Motorcar = st.selectbox("Household has Motorcar/Jeep/Van?", options=[0, 1], index=0)
Is_HH_Have_Trucks = st.selectbox("Household has Trucks?", options=[0, 1], index=0)
Is_HH_Have_Animal_cart = st.selectbox("Household has Animal cart?", options=[0, 1], index=0)
Is_HH_Have_Refrigerator = st.selectbox("Household has Refrigerator?", options=[0, 1], index=1)
Is_HH_Have_Washing_machine = st.selectbox("Household has Washing Machine?", options=[0, 1], index=0)
Is_HH_Have_Airconditioner = st.selectbox("Household has Airconditioner/Aircooler?", options=[0, 1], index=0)

# ===================================================
# PERSON-LEVEL INPUTS
# ===================================================

st.header("Person-Level Information")
num_persons = st.number_input("Enter number of household members", min_value=1, value=3, step=1)

# List to store each person's input data.
persons_data = []

# For each person, ask for their details in an expander.
for i in range(int(num_persons)):
    with st.expander(f"Enter details for Person {i+1}"):
        age = st.number_input(f"Age (in years) for Person {i+1}", min_value=0, value=30, key=f"age_{i}")
        gender = st.selectbox(f"Gender for Person {i+1}", options=["Male", "Female"], key=f"gender_{i}")
        # Map gender to code: Male=1, Female=2.
        gender_code = 1 if gender == "Male" else 2
        marital_status = st.selectbox(f"Marital Status for Person {i+1}", options=["Not Married", "Married"], key=f"marital_{i}")
        marital_code = 2 if marital_status == "Married" else 1
        education = st.number_input(f"Highest Educational Level (code) for Person {i+1}", min_value=0, value=4, key=f"edu_{i}")
        internet_usage = st.selectbox(f"Used Internet in last 30 days for Person {i+1}?", options=["No", "Yes"], key=f"internet_{i}")
        internet_code = 1 if internet_usage == "Yes" else 0
        # Meals information
        meals_school = st.number_input(f"Meals from school for Person {i+1}", min_value=0, value=0, key=f"meals_school_{i}")
        meals_employer = st.number_input(f"Meals from employer for Person {i+1}", min_value=0, value=0, key=f"meals_employer_{i}")
        meals_payment = st.number_input(f"Meals on payment for Person {i+1}", min_value=0, value=0, key=f"meals_payment_{i}")
        meals_home = st.number_input(f"Meals at home for Person {i+1}", min_value=0, value=0, key=f"meals_home_{i}")
        meals_others = st.number_input(f"Meals from others for Person {i+1}", min_value=0, value=0, key=f"meals_others_{i}")

        persons_data.append({
            "Age(in years)": age,
            "Gender": gender_code,
            "Marital Status (code)": marital_code,
            "Highest educational level attained (code)": education,
            "Whether used internet from any location during last 30 days": internet_code,
            "No. of meals taken during last 30 days from school, balwadi etc.": meals_school,
            "No. of meals taken during last 30 days from employer as perquisites or part of wage": meals_employer,
            "No. of meals taken during last 30 days on payment": meals_payment,
            "No. of meals taken during last 30 days at home": meals_home,
            "No. of meals taken during last 30 days  others": meals_others
        })

# Convert person-level inputs into a DataFrame.
df_persons = pd.DataFrame(persons_data)

# ===================================================
# CALCULATE AGGREGATED PERSON-LEVEL FEATURES
# ===================================================

# Count of adults: Age 18-64 (inclusive)
adults_df = df_persons[(df_persons["Age(in years)"] >= 18) & (df_persons["Age(in years)"] <= 64)]
count_adults = len(adults_df)
adults_mean_age = adults_df["Age(in years)"].mean() if count_adults > 0 else 0

# Count of children: Age below 18
children_df = df_persons[df_persons["Age(in years)"] < 18]
count_children = len(children_df)
children_mean_age = children_df["Age(in years)"].mean() if count_children > 0 else 0

# Count of elders: Age 65 and above
elders_df = df_persons[df_persons["Age(in years)"] >= 65]
count_elders = len(elders_df)
elders_mean_age = elders_df["Age(in years)"].mean() if count_elders > 0 else 0

# Gender counts and ratio
count_males = len(df_persons[df_persons["Gender"] == 1])
count_females = len(df_persons[df_persons["Gender"] == 2])
gender_ratio = count_females / (count_males + 1)
mean_age = df_persons["Age(in years)"].mean()

# Dependency ratio: (children + elders) / (adults + 1)
dependency_ratio = (count_children + count_elders) / (count_adults + 1)

# Count of married persons (assuming marital code 2 indicates married)
count_married = len(df_persons[df_persons["Marital Status (code)"] == 2])
# Maximum educational level attained among persons
max_education = df_persons["Highest educational level attained (code)"].max()
# Count of internet users (where code == 1)
count_internet_users = len(df_persons[df_persons["Whether used internet from any location during last 30 days"] == 1])
# Sum up meals from various sources
total_meals_school = df_persons["No. of meals taken during last 30 days from school, balwadi etc."].sum()
total_meals_employer = df_persons["No. of meals taken during last 30 days from employer as perquisites or part of wage"].sum()
total_meals_payment = df_persons["No. of meals taken during last 30 days on payment"].sum()
total_meals_home = df_persons["No. of meals taken during last 30 days at home"].sum()
total_meals_others = df_persons["No. of meals taken during last 30 days  others"].sum()

st.markdown("#### Aggregated Person-Level Features")
st.write(f"Count of Adults: {count_adults}")
st.write(f"Adults Mean Age: {adults_mean_age:.2f}")
st.write(f"Count of Children: {count_children}")
st.write(f"Children Mean Age: {children_mean_age:.2f}")
st.write(f"Count of Elders: {count_elders}")
st.write(f"Elders Mean Age: {elders_mean_age:.2f}")
st.write(f"Gender Ratio (Females/Males): {gender_ratio:.2f}")
st.write(f"Overall Mean Age: {mean_age:.2f}")
st.write(f"Dependency Ratio: {dependency_ratio:.2f}")
st.write(f"Count Married: {count_married}")
st.write(f"Max Education: {max_education}")
st.write(f"Count Internet Users: {count_internet_users}")
st.write(f"Total Meals (School): {total_meals_school}")
st.write(f"Total Meals (Employer): {total_meals_employer}")
st.write(f"Total Meals (Payment): {total_meals_payment}")
st.write(f"Total Meals (Home): {total_meals_home}")
st.write(f"Total Meals (Others): {total_meals_others}")

# ===================================================
# ASSEMBLE FINAL INPUT FOR MODEL
# ===================================================

# Create a DataFrame with all required features.
input_data = pd.DataFrame({
    "Sector": [Sector],
    "State": [State],
    "NSS-Region": [nss_region],
    "District": [District],
    "Household Type": [Household_Type],
    "Religion of the head of the household": [Religion],
    "Social Group of the head of the household": [Social_Group],
    "HH Size (For FDQ)": [HH_Size],
    "NCO_3D": [NCO_3D],
    "NIC_5D": [NIC_5D],
    "Is_online_Clothing_Purchased_Last365": [Is_online_Clothing],
    "Is_online_Footwear_Purchased_Last365": [Is_online_Footwear],
    "Is_online_Furniture_fixturesPurchased_Last365": [Is_online_Furniture],
    "Is_online_Mobile_Handset_Purchased_Last365": [Is_online_Mobile],
    "Is_online_Personal_Goods_Purchased_Last365": [Is_online_Personal_Goods],
    "Is_online_Recreation_Goods_Purchased_Last365": [Is_online_Recreation],
    "Is_online_Household_Appliances_Purchased_Last365": [Is_online_Household_Appliances],
    "Is_online_Crockery_Utensils_Purchased_Last365": [Is_online_Crockery],
    "Is_online_Sports_Goods_Purchased_Last365": [Is_online_Sports],
    "Is_online_Medical_Equipment_Purchased_Last365": [Is_online_Medical],
    "Is_online_Bedding_Purchased_Last365": [Is_online_Bedding],
    "Is_HH_Have_Television": [Is_HH_Have_Television],
    "Is_HH_Have_Radio": [Is_HH_Have_Radio],
    "Is_HH_Have_Laptop_PC": [Is_HH_Have_Laptop_PC],
    "Is_HH_Have_Mobile_handset": [Is_HH_Have_Mobile_handset],
    "Is_HH_Have_Bicycle": [Is_HH_Have_Bicycle],
    "Is_HH_Have_Motorcycle_scooter": [Is_HH_Have_Motorcycle_scooter],
    "Is_HH_Have_Motorcar_jeep_van": [Is_HH_Have_Motorcar],
    "Is_HH_Have_Trucks": [Is_HH_Have_Trucks],
    "Is_HH_Have_Animal_cart": [Is_HH_Have_Animal_cart],
    "Is_HH_Have_Refrigerator": [Is_HH_Have_Refrigerator],
    "Is_HH_Have_Washing_machine": [Is_HH_Have_Washing_machine],
    "Is_HH_Have_Airconditioner_aircooler": [Is_HH_Have_Airconditioner],
    "count_adults": [count_adults],
    "adults_mean_age": [adults_mean_age],
    "count_children": [count_children],
    "children_mean_age": [children_mean_age],
    "count_elders": [count_elders],
    "elders_mean_age": [elders_mean_age],
    "count_males": [count_males],
    "count_females": [count_females],
    "gender_ratio": [gender_ratio],
    "mean_age": [mean_age],
    "dependency_ratio": [dependency_ratio],
    "count_married": [count_married],
    "max_education": [max_education],
    "count_internet_users": [count_internet_users],
    "total_meals_school": [total_meals_school],
    "total_meals_employer": [total_meals_employer],
    "total_meals_payment": [total_meals_payment],
    "total_meals_home": [total_meals_home],
    "total_meals_others": [total_meals_others]
})

# ===================================================
# PREDICTION
# ===================================================

if st.button("Predict Total Expense"):
    # Model was trained on log-transformed target values; apply inverse transform.
    y_pred_log = model.predict(input_data)
    y_pred = np.expm1(y_pred_log)
    
    st.success(f"Predicted Total Expense: {y_pred[0]:.2f}")
