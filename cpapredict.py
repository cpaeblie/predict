import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from numpy import array

# Load dataset
zymuno_df = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/main/ad%20final.csv', delimiter=',')
df_ori = zymuno_df
df_ori['Date'] = pd.to_datetime(df_ori['Date'])
df_X = df_ori[['Cost', 'CPC (Destination)', 'CPM', 'Impression', 'Clicks (Destination)', 'CTR (Destination)', 'Conversions', 'CPA']]
in_seq = df_X.astype(float).values

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
    return array(X), array(y)

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

# Prepare data for training
n_steps_in, n_steps_out = 4, 1
X, y = split_sequences(in_seq, n_steps_in, n_steps_out)
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
X_train = stats_features(X_train)
X_test = stats_features(X_test)

# Streamlit application
st.set_page_config(page_title="CPA Prediction App", page_icon="🔎")

# Sidebar menu layout
st.sidebar.title("Menu")
if st.sidebar.button("Prediction"):
    menu = "Prediction"
elif st.sidebar.button("History"):
    menu = "History"
elif st.sidebar.button("Dataset"):
    menu = "Dataset"
else:
    menu = "Prediction"  # Default to Prediction if no button is clicked

if menu == "Prediction":
    st.title("CPA Prediction App 🔎")
    st.write("""
    This application predicts the Cost Per Acquisition (CPA) based on input features.
    """)
    
    new_name_inputs = []
    with st.form("cpa_form"):
        for i in range(16):
            day = (i // 4) + 1
            metric = ["Cost", "CPC (Destination)", "CPM", "CTR (Destination)"][i % 4]
            new_name_input = st.text_input(label=f'{metric} at Day {day}:', key=f'input_{i+16}')
            new_name_inputs.append(new_name_input)
        if st.form_submit_button("Predict The CPA!"):
            new_name = np.array([float(new_name_input) for new_name_input in new_name_inputs]).reshape(-1, X_test.shape[1])
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(new_name)
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            st.sidebar.write("Tomorrow's CPA Prediction:")
            st.sidebar.write(np.round(y_pred, 0))

    st.write("""
    Please refresh the website if you want to input new values.
    """)

elif menu == "History":
    st.title("History of Marketing Metrics")
    
    # Create line charts for each metric
    st.subheader("Cost")
    st.line_chart(df_ori.set_index('Date')['Cost'])

    st.subheader("CPC (Destination)")
    st.line_chart(df_ori.set_index('Date')['CPC (Destination)'])

    st.subheader("CPM")
    st.line_chart(df_ori.set_index('Date')['CPM'])

    st.subheader("Impression")
    st.line_chart(df_ori.set_index('Date')['Impression'])

    st.subheader("Clicks (Destination)")
    st.line_chart(df_ori.set_index('Date')['Clicks (Destination)'])

    st.subheader("CTR (Destination)")
    st.line_chart(df_ori.set_index('Date')['CTR (Destination)'])

    st.subheader("Conversions")
    st.line_chart(df_ori.set_index('Date')['Conversions'])

    st.subheader("CPA")
    st.line_chart(df_ori.set_index('Date')['CPA'])

elif menu == "Dataset":
    st.title("Dataset")
    st.write(df_ori)

st.caption('Copyright (c) PT Ebliethos Indonesia 2024')
