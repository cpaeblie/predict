import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import kurtosis, skew

# User registration and login management
users_db = {}

def register_user(username, password):
    if username in users_db:
        return False
    users_db[username] = password
    return True

def login_user(username, password):
    return users_db.get(username) == password

# Data processing functions
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[out_end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def stats_features(input_data):
    inp = []
    for i in range(len(input_data)):
        inp2 = input_data[i]
        features = [
            np.min(inp2), np.max(inp2), 
            np.max(inp2) - np.min(inp2), 
            np.std(inp2), np.mean(inp2), 
            np.median(inp2), kurtosis(inp2), 
            skew(inp2)
        ]
        inp.append(np.concatenate((inp2, features)))
    return np.array(inp)

# Load dataset
zymuno_df = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/main/ad%20final.csv', delimiter=',')
zymuno_df['Date'] = pd.to_datetime(zymuno_df['Date'])
df_X = zymuno_df[['Cost', 'CPC (Destination)', 'CPM', 'CTR (Destination)', 'CPA']].astype(float).values

# Streamlit app setup
st.set_page_config(page_title="CPA Prediction App", page_icon="🔎")

# User authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    user_name = st.session_state.user_name
else:
    st.sidebar.title("Login/Register")
    option = st.sidebar.selectbox("Select an option", ["Login", "Register"])
    
    if option == "Register":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Register"):
            if register_user(username, password):
                st.success("Registered successfully!")
            else:
                st.error("Username already exists.")
    
    elif option == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.user_name = username
                st.success(f"Welcome {username}!")
            else:
                st.error("Invalid credentials.")

# Main app logic for logged-in users
if st.session_state.logged_in:
    st.sidebar.title("Dashboard")
    menu = st.sidebar.radio("Select Menu", ["History", "Dataset", "Recent Page"])
    
    # History Menu
    if menu == "History":
        st.subheader("History")
        # Display line chart of each column in the dataset
        for column in df_X.columns:
            st.line_chart(zymuno_df[column])
    
    # Dataset Menu
    elif menu == "Dataset":
        st.subheader("Dataset")
        st.dataframe(zymuno_df)
    
    # Recent Page for CPA Prediction
    elif menu == "Recent Page":
        st.subheader("CPA Prediction")
        
        # Prepare input form
        new_name_inputs = []
        with st.form("cpa_form"):
            for i in range(16):
                day = (i // 4) + 1
                metric = ["Cost", "CPC (Destination)", "CPM", "CTR (Destination)"][i % 4]
                new_name_input = st.text_input(label=f'{metric} at Day {day}:', key=f'input_{i+16}')
                new_name_inputs.append(new_name_input)
            if st.form_submit_button("Predict The CPA!"):
                new_name = np.array([float(new_name_input) for new_name_input in new_name_inputs]).reshape(-1, X.shape[1])
                
                # Data processing for prediction
                n_steps_in, n_steps_out = 4, 1
                X, y = split_sequences(df_X, n_steps_in, n_steps_out)

                # Prepare the training and testing datasets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
                X_train = stats_features(X_train)
                X_test = stats_features(X_test)

                # Handle missing values
                imputer = SimpleImputer(strategy='mean')
                X_train_imputed = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test)

                # Remove NaN values
                X_train_no_nan = X_train_imputed[~np.isnan(X_train_imputed).any(axis=1)]
                X_test_no_nan = X_test_imputed[~np.isnan(X_test_imputed).any(axis=1)]
                y_train_no_nan = y_train[~np.isnan(y_train)]
                y_test_no_nan = y_test[~np.isnan(y_test)]

                # Scale the input features
                scaler = StandardScaler().fit(X_train_no_nan)
                X_train_scaled = scaler.transform(X_train_no_nan)
                X_test_scaled = scaler.transform(new_name)

                # Hyperparameter tuning and model fitting
                param_dist = {
                    'n_estimators': [10, 50, 100, 200, 500],
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [2, 5, 10, 20, 30],
                    'min_samples_leaf': [1, 2, 4, 8, 16]
                }
                
                model = RandomForestRegressor(random_state=42)
                random_search = RandomizedSearchCV(estimator=model, 
                                                   param_distributions=param_dist, 
                                                   cv=5, 
                                                   scoring='neg_mean_squared_error', 
                                                   verbose=0, 
                                                   n_iter=20, 
                                                   random_state=42)
                random_search.fit(X_train_scaled, y_train_no_nan)

                best_model = random_search.best_estimator_
                best_model.fit(X_train_scaled, y_train_no_nan)

                # Make predictions
                y_pred = best_model.predict(X_test_scaled)
                y_pred = np.round(y_pred, 0)

                # Display the predictions
                st.sidebar.write("Tomorrow's CPA Prediction:")
                st.sidebar.write(y_pred)

    # Display username and logout option
    st.sidebar.write(f"Logged in as: {user_name}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        del st.session_state.user_name
        st.success("Logged out successfully!")

st.caption('Copyright (c) PT Ebliethos Indonesia 2024')
