import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

# Set up the app
st.set_page_config(page_title="CPA Prediction App", page_icon="🔎", layout="wide")

# Authentication system
users_db = {"admin": "password"}  # Simple in-memory user "database" for demo

# Function for login page
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users_db and users_db[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

# Function for registration page
def register():
    st.title("Register")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Register"):
        if username not in users_db:
            users_db[username] = password
            st.success("User registered successfully! You can now log in.")
            st.experimental_rerun()
        else:
            st.error("Username already exists")

# Load data
zymuno_df = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/main/ad%20final.csv')
zymuno_df['Date'] = pd.to_datetime(zymuno_df['Date'])
df_X = zymuno_df[['Cost', 'CPC (Destination)', 'CPM', 'CTR (Destination)', 'CPA']]

# Splitting sequences (your existing function)
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

# Statistics features (your existing function)
def stats_features(input_data):
    inp = []
    for inp2 in input_data:
        features = [
            np.min(inp2), np.max(inp2), np.ptp(inp2),
            np.std(inp2), np.mean(inp2), np.median(inp2),
            kurtosis(inp2), skew(inp2)
        ]
        inp.append(np.append(inp2, features))
    return np.array(inp)

# Page layout and content
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["History", "Dataset", "Prediksi"])

    # User info & logout
    st.sidebar.markdown("---")
    st.sidebar.write(f"Welcome, **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()

    # Page content
    if selected_page == "History":
        st.title("History")
        st.write("Line chart of each column in the dataset")
        
        for column in df_X.columns:
            st.write(f"Line chart for **{column}**")
            fig, ax = plt.subplots()
            ax.plot(zymuno_df["Date"], df_X[column])
            ax.set_xlabel("Date")
            ax.set_ylabel(column)
            st.pyplot(fig)

    elif selected_page == "Dataset":
        st.title("Dataset")
        st.write("Displaying the dataset in a table format")
        st.dataframe(zymuno_df)

    elif selected_page == "Prediksi":
        st.title("Prediksi")
        st.write("Enter the values for 4 days to predict CPA")

        n_steps_in, n_steps_out = 4, 1
        X, y = split_sequences(df_X.values, n_steps_in, n_steps_out)
        n_input = X.shape[1] * X.shape[2]
        X = X.reshape((X.shape[0], n_input))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = stats_features(X_train)
        X_test = stats_features(X_test)

        new_inputs = []
        for day in range(1, 5):
            st.write(f"Day {day}")
            for feature in ["Cost", "CPC (Destination)", "CPM", "CTR (Destination)"]:
                val = st.number_input(f"{feature} on Day {day}", min_value=0.0)
                new_inputs.append(val)

        if st.button("Predict The CPA!"):
            new_input = np.array(new_inputs).reshape(1, -1)
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            new_input_scaled = scaler.transform(new_input)

            param_dist = {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }

            model = RandomForestRegressor(random_state=42)
            random_search = RandomizedSearchCV(
                model, param_dist, cv=5, n_iter=5, random_state=42, scoring='neg_mean_squared_error'
            )
            random_search.fit(X_train_scaled, y_train)
            best_model = random_search.best_estimator_

            y_pred = best_model.predict(new_input_scaled)
            st.write(f"Predicted CPA for tomorrow: {round(y_pred[0], 2)}")

else:
    page_choice = st.selectbox("Choose an option", ["Login", "Register"])
    if page_choice == "Login":
        login()
    elif page_choice == "Register":
        register()
