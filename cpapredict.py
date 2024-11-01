import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
import streamlit as st
from scipy.stats import kurtosis, skew

# Function to split sequences
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

# Function to extract statistical features
def stats_features(input_data):
    inp = list()
    for i in range(len(input_data)):
        inp2 = input_data[i]
        min_val = float(np.min(inp2))
        max_val = float(np.max(inp2))
        diff = (max_val - min_val)
        std = float(np.std(inp2))
        mean = float(np.mean(inp2))
        median = float(np.median(inp2))
        kurt = float(kurtosis(inp2))
        sk = float(skew(inp2))
        inp2 = np.append(inp2, [min_val, max_val, diff, std, mean, median, kurt, sk])
        inp.append(inp2)
    return np.array(inp)

# Load dataset
zymuno_df = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/main/ad%20final.csv', delimiter=',')
df_ori = zymuno_df
df_ori['Date'] = pd.to_datetime(df_ori['Date'])
df_X = df_ori[['Cost', 'CPC (Destination)', 'CPM', 'CTR (Destination)', 'CPA']]
in_seq = df_X.astype(float).values

# Prepare data for model
n_steps_in, n_steps_out = 4, 1
X, y = split_sequences(in_seq, n_steps_in, n_steps_out)
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
X_train = stats_features(X_train)
X_test = stats_features(X_test)

# Initialize Streamlit app
st.set_page_config(page_title="CPA Prediction App", page_icon="🔎")
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select a page:", ["Prediction", "Dataset", "History"])

if menu == "Prediction":
    
    # Prediction Page
    st.title("CPA Prediction App 🔎")
    st.write("This is a CPA Prediction App that uses machine learning algorithms to predict the Cost Per Acquisition (CPA) for a given set of input features.")
    
    new_name_inputs = []
    with st.form("cpa_form"):
        for i in range(16):
            day = (i // 4) + 1
            metric = ["Cost", "CPC (Destination)", "CPM", "CTR (Destination)"][i % 4]
            new_name_input = st.text_input(label=f'{metric} at Day {day}:', key=f'input_{i+16}')
            new_name_inputs.append(new_name_input)
        
        if st.form_submit_button("Predict The CPA!"):
            new_name = np.array([float(new_name_input) for new_name_input in new_name_inputs]).reshape(-1, X_test.shape[1])
            
            # Scale the input features
            scaler = StandardScaler().fit(X_train)
            X_test_scaled = scaler.transform(new_name)

            # Hyperparameter tuning for Random Forest
            param_dist = {
                'n_estimators': [10, 50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10, 20, 30],
                'min_samples_leaf': [1, 2, 4, 8, 16]
            }
            model = RandomForestRegressor(random_state=42)

            # Perform hyperparameter tuning using RandomizedSearchCV
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=5, scoring='neg_mean_squared_error', verbose=0, n_iter=20, random_state=42)
            random_search.fit(X_train, y_train)

            # Extract the best model and fit it to the training data
            best_model = random_search.best_estimator_
            best_model.fit(X_train, y_train)

            # Make predictions on the new input
            y_pred = best_model.predict(X_test_scaled)
            y_pred = np.round(y_pred, 0)

            # Display the predictions below the button
            st.write("Tomorrow's CPA Prediction:")
            st.write(y_pred)

    st.write("Please refresh the website if you want to input new values.")




elif menu == "Dataset":
    
    # Dataset Page
    st.title("Dataset")
    st.write("Here is the dataset used for the CPA prediction.")
    st.dataframe(df_ori)


elif menu == "History":
    
    # History Page
    st.title("History")
    st.write("This section will display the line charts of each column in the dataset.")
    
    # Plotting the line charts for each column
    st.subheader("Line Charts")
    
    # Date vs CPA
    st.line_chart(df_ori.set_index('Date')['CPA'], use_container_width=True)
    
    # Date vs Cost
    st.line_chart(df_ori.set_index('Date')['Cost'], use_container_width=True)
    
    # Date vs CPC (Destination)
    st.line_chart(df_ori.set_index('Date')['CPC (Destination)'], use_container_width=True)

    # Date vs CPM
    st.line_chart(df_ori.set_index('Date')['CPM'], use_container_width=True)

    # Date vs CTR (Destination)
    st.line_chart(df_ori.set_index('Date')['CTR (Destination)'], use_container_width=True)


st.caption('Copyright (c) PT Ebliethos Indonesia 2024')
