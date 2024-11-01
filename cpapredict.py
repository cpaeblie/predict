import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import kurtosis, skew
import hashlib

# **User  Authentication Functions**

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(hashed_password, user_password):
    return hashed_password == hash_password(user_password)

# **User  Registration and Login**

users_db = {}  # This will act as a simple in-memory database

def register_user(username, password):
    if username in users_db:
        return False
    users_db[username] = hash_password(password)
    return True

def login_user(username, password):
    if username in users_db and check_password(users_db[username], password):
        return True
    return False

# **Streamlit App Setup**

st.set_page_config(page_title="CPA Prediction App", page_icon="🔎")
st.title("CPA Prediction App 🔎")

# **User  Registration/Login Section**

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Register"):
        if register_user(username, password):
            st.success("Registered successfully! You can now log in.")
        else:
            st.error("Username already exists.")

    st.subheader("Login")
    login_username = st.text_input("Username", key="login_username")
    login_password = st.text_input("Password", type='password', key="login_password")
    
    if st.button("Login"):
        if login_user(login_username, login_password):
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")

else:
    # **CPA Prediction Logic**

    st.write("""
    This is a CPA Prediction App that uses machine learning algorithms to predict the Cost Per Acquisition (CPA) for a given set of input features (Cost, CPC (Destination), CPM, CTR (Destination), CPA) for the 4 days before tomorrow.
    """)

    # **Input Features**
    new_name_inputs = []
    with st.form("cpa_form"):
        for i in range(16):
            day = (i // 4) + 1
            metric = ["Cost", "CPC (Destination)", "CPM", "CTR (Destination)"][i % 4]
            new_name_input = st.text_input(label=f'{metric} at Day {day}:', key=f'input_{i+16}')
            new_name_inputs.append(new_name_input)
        if st.form_submit_button("Predict The CPA!"):
            # Load data and prepare for prediction
            zymuno_df = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/main/ad%20final.csv', delimiter=',')
            df_ori = zymuno_df
            df_ori['Date'] = pd.to_datetime(df_ori['Date'])
            df_X = df_ori[['Cost','CPC (Destination)','CPM','CTR (Destination)','CPA']]
            in_seq = df_X.astype(float).values
            
            n_steps_in, n_steps_out = 4, 1
            X, y = split_sequences(in_seq, n_steps_in, n_steps_out)

            n_input = X.shape[1] * X.shape[2]
            X = X.reshape((X.shape[0], n_input))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
            X_train = stats_features(X_train)
            X_test = stats_features(X_test)

            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)

            # Scale the input features
            scaler = StandardScaler().fit(X_train_imputed)
            new_input = np.array([float(input_value) for input_value in new_name_inputs]).reshape(1, -1)
            new_input_scaled = scaler.transform(new_input)

            # Train model
            param_dist = {
                'n_estimators': [10, 50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10, 20, 30],
		                'min_samples_leaf': [1, 2, 4, 8, 16]
            }

            # Initialize the Random Forest Regressor model
            model = RandomForestRegressor(random_state=42)

            # Perform hyperparameter tuning using RandomizedSearchCV
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=5,
                                               scoring='neg_mean_squared_error', verbose=0, n_iter=20, random_state=42)
            random_search.fit(X_train_imputed, y_train)

            # Extract the best model and fit it to the training data
            best_model = random_search.best_estimator_
            best_model.fit(X_train_imputed, y_train)

            # Make predictions on the new input data
            y_pred = best_model.predict(new_input_scaled)
            y_pred = np.round(y_pred, 0)

            # Display the predictions in the sidebar
            st.sidebar.write("Tomorrow's CPA Prediction:")
            st.sidebar.write(y_pred[0])

    st.write("""
    Please refresh the website if you want to input new values
    """)

# **Footer Section**
st.caption('Copyright (c) PT Ebliethos Indonesia 2024')
