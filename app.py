import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# App Title
st.title("🔥 **NextGen Fitness Tracker**")
st.write(
    """
    Welcome to your personal fitness assistant! 🌟 
    This tool estimates **calories burned** based on key fitness metrics. 
    Get ready to monitor and optimize your health journey! 💪
    """
)

# Sidebar Header
st.sidebar.header("Enter Your Fitness Details:")

# User Input Features
def fetch_user_inputs():
    st.sidebar.subheader("Provide Your Details Below")
    age = st.sidebar.slider("Age (years):", 10, 100, 25)
    bmi = st.sidebar.slider("Body Mass Index (BMI):", 15.0, 50.0, 22.0)
    duration = st.sidebar.slider("Exercise Time (minutes):", 0, 180, 45)
    heart_rate = st.sidebar.slider("Heart Rate (BPM):", 50, 200, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C):", 35.0, 42.0, 37.5)
    gender = st.sidebar.radio("Gender:", ("Male", "Female"))

    # Encode gender as numeric for the model
    gender_binary = 1 if gender == "Male" else 0

    # Return input as a DataFrame
    inputs = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_Male": gender_binary
    }
    return pd.DataFrame(inputs, index=[0])

user_inputs = fetch_user_inputs()

st.write("### Your Input Data:")
st.write(user_inputs)

# Simulate Progress Bar
with st.spinner("Crunching the numbers... Hang tight!"):
    time.sleep(1.5)

# Load and Validate Dataset
try:
    calories_data = pd.read_csv("calories.csv")
    exercise_data = pd.read_csv("exercise.csv")
except FileNotFoundError:
    st.error("⚠️ Missing dataset files (`calories.csv` and `exercise.csv`). Please add them and reload.")
    st.stop()

# Data Preprocessing
combined_data = exercise_data.merge(calories_data, on="User_ID", how="inner").drop(columns=["User_ID"])
combined_data["BMI"] = combined_data["Weight"] / ((combined_data["Height"] / 100) ** 2)

# Splitting Data for Training
train_set, test_set = train_test_split(combined_data, test_size=0.2, random_state=42)
X_train = pd.get_dummies(train_set.drop(columns="Calories"), drop_first=True)
y_train = train_set["Calories"]
X_test = pd.get_dummies(test_set.drop(columns="Calories"), drop_first=True)
y_test = test_set["Calories"]

# Model Training
rf_model = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)

# Prediction Based on User Inputs
user_inputs = user_inputs.reindex(columns=X_train.columns, fill_value=0)
calories_prediction = rf_model.predict(user_inputs)[0]

st.write("### Estimated Calories Burned:")
st.success(f"🔥 **{round(calories_prediction, 2)} kcal** burned!")

# Similar Data Insights
st.write("### Insights from Similar Fitness Records:")
calorie_limits = [calories_prediction - 50, calories_prediction + 50]
similar_records = combined_data[
    (combined_data["Calories"] >= calorie_limits[0]) & (combined_data["Calories"] <= calorie_limits[1])
]

# Display Insights
if not similar_records.empty:
    st.write("Here are similar fitness records from other users:")
    st.dataframe(similar_records.sample(5))
    st.write(f"💡 Average Calories: {similar_records['Calories'].mean():.2f}")
    st.write(f"💪 Highest Calories: {similar_records['Calories'].max():.2f}")
    st.write(f"🧘 Lowest Calories: {similar_records['Calories'].min():.2f}")
else:
    st.write("No similar records found. Try different input values.")

# Data Visualization
st.write("### Exercise Duration vs Calories Burned:")
st.line_chart(similar_records[["Duration", "Calories"]].set_index("Duration"))

