import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Simulated user database
user_db = {}

# Function for user registration
def register_user(username, password):
    if username in user_db:
        return False
    user_db[username] = password
    return True

# Function for user login
def login_user(username, password):
    return user_db.get(username) == password

# Load dataset
def load_data():
    zymuno_df = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/main/ad%20final.csv', delimiter=',')
    zymuno_df['Date'] = pd.to_datetime(zymuno_df['Date'])
    return zymuno_df

# Function to display dataset
def display_dataset(df):
    st.write(df)

# Function to plot history
def plot_history(df):
    st.line_chart(df.set_index('Date'))

# Main application logic
st.set_page_config(page_title="CPA Prediction App", page_icon="🔎")
st.title("CPA Prediction App 🔎")

menu = st.sidebar.selectbox("Select Menu", ["Login", "Register", "Dashboard"])

if menu == "Register":
    st.subheader("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Register"):
        if register_user(username, password):
            st.success("User  registered successfully!")
        else:
            st.warning("Username already exists.")

elif menu == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if login_user(username, password):
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
        else:
            st.warning("Incorrect username or password.")

elif menu == "Dashboard":
    if st.session_state.get("logged_in"):
        st.subheader("Dashboard")
        df = load_data()
        
        # Display History
        st.header("History")
        plot_history(df[['Date', 'CPA']])
        
        # Display Dataset
        st.header("Dataset")
        display_dataset(df)

        # Recent Page for CPA Prediction
        st.header("CPA Prediction")
        new_name_inputs = []
        with st.form("cpa_form"):
            for i in range(32):
                day = (i // 8) + 1
                metric = ["Cost", "CPC (Destination)", "CPM", "Impression", 
                          "Clicks (Destination)", "CTR (Destination)", 
                          "Conversions", "CPA"][i % 8]
                new_name_input = st.text_input(label=f'{metric} at Day {day}:', key=f'input_{i+32}')
                new_name_inputs.append(new_name_input)
            if st.form_submit_button("Predict The CPA!"):
                new_name = np.array([float(new_name_input) for new_name_input in new_name_inputs]).reshape(1, -1)
                
                # Preprocessing
                X = df[['Cost', 'CPC (Destination)', 'CPM', 'Impression', 
                         'Clicks (Destination)', 'CTR (Destination)', 
                         'Conversions']].values
                y = df['CPA'].values

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                imputer = SimpleImputer(strategy='mean')
                X_train_imputed = imputer.fit_transform(X_train)
                scaler = StandardScaler().fit(X_train_imputed)
                X_train_scaled = scaler.transform(X_train_imputed)
                
                                # Model Training
                model = RandomForestRegressor(random_state=42)
                param_dist = {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
                
                # Perform hyperparameter tuning using RandomizedSearchCV
                random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=5)
                random_search.fit(X_train_scaled, y_train)

                # Extract the best model and fit it to the training data
                best_model = random_search.best_estimator_
                best_model.fit(X_train_scaled, y_train)

                # Scale the new input features
                new_name_scaled = scaler.transform(new_name)

                # Make predictions
                y_pred = best_model.predict(new_name_scaled)
                y_pred = np.round(y_pred, 0)

                # Display the predictions
                st.sidebar.write("Tomorrow's CPA Prediction:")
                st.sidebar.write(y_pred)

    else:
        st.warning("Please log in to access the dashboard.")

# Footer
st.caption('Copyright (c) PT Ebliethos Indonesia 2024')
