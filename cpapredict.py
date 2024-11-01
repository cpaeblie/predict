import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer
import dotenv
from streamlit_supabase_auth import login_form, logout_button
from supabase import create_client, Client
import os

# Load environment variables
dotenv.load_dotenv()

# Set up Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure Streamlit app
st.set_page_config(page_title="CPA Prediction App", page_icon="🔎")
st.title("CPA Prediction App 🔎")
st.write("""
This is a CPA Prediction App that uses machine learning algorithms to predict the Cost Per Acquisition (CPA).
""")

# User Authentication
session = login_form(
    url=SUPABASE_URL,
    apiKey=SUPABASE_KEY,
    providers=["email", "github", "google"],
)

# If the user is not logged in, stop the app
if not session:
    st.stop()

# Display a welcome message and logout button
with st.sidebar:
    st.write(f"Welcome, {session['user']['email']}")
    logout_button()

# Sidebar with additional information
st.sidebar.title("App Info")
st.sidebar.info(
    """
    CPA Prediction App uses machine learning to predict future CPA.
    Developed by PT Ebliethos Indonesia, 2024.
    """
)

# Load and preprocess the data
zymuno_df = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/main/ad%20final.csv', delimiter=',')
df_ori = zymuno_df
df_ori['Date'] = pd.to_datetime(df_ori['Date'])
df_X = df_ori[['Cost','CPC (Destination)','CPM','CTR (Destination)','CPA']]
in_seq = df_X.astype(float).values

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

n_steps_in, n_steps_out = 4, 1
X, y = split_sequences(in_seq, n_steps_in, n_steps_out)
X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define a form for user input
new_name_inputs = []
with st.form("cpa_form"):
    st.write("Enter metrics for Day 1 to Day 4:")
    for i in range(16):
        day = (i // 4) + 1
        metric = ["Cost", "CPC (Destination)", "CPM", "CTR (Destination)"][i % 4]
        new_name_input = st.text_input(label=f'{metric} at Day {day}:', key=f'input_{i}')
        new_name_inputs.append(new_name_input)

    if st.form_submit_button("Predict The CPA!"):
        # Convert input values to float and reshape for model input
        new_name = np.array([float(new_name_input) for new_name_input in new_name_inputs]).reshape(-1, X_test.shape[1])

        # Scale the input features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(new_name)

        # Define the hyperparameter distribution
        param_dist = {
            'n_estimators': [10, 50, 100, 200, 500],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20, 30],
            'min_samples_leaf': [1, 2, 4, 8, 16]
        }

        # Initialize the Random Forest Regressor model and perform hyperparameter tuning
        model = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=5, 
                                           scoring='neg_mean_squared_error', n_iter=20, random_state=42)
        random_search.fit(X_train_scaled, y_train)

        # Get the best model and make predictions
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        y_pred = np.round(y_pred, 0)

        # Display the predictions in the sidebar
        st.sidebar.write("Tomorrow's CPA Prediction:")
        st.sidebar.write(y_pred)

st.write("Please refresh the website to input new values")
st.caption('Copyright (c) PT Ebliethos Indonesia 2024')
