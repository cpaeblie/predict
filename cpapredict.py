import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import streamlit as st
from scipy.stats import kurtosis, skew
from numpy import array

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
        inp = np.append(inp, inp2)
    inp = inp.reshape(len(input_data), -1)
    return inp

# Load dataset
zymuno_df = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/main/ad%20final.csv', delimiter=',')
df_ori = zymuno_df
df_ori['Date'] = pd.to_datetime(df_ori['Date'])
df_X = df_ori[['Cost', 'CPC (Destination)', 'CPM', 'CTR (Destination)', 'CPA']]
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

X_train_no_nan = X_train[~np.isnan(X_train).any(axis=1)]
X_test_no_nan = X_test[~np.isnan(X_test).any(axis=1)]
y_train_no_nan = y_train[~np.isnan(y_train)]
y_test_no_nan = y_test[~np.isnan(y_test)]

# Streamlit app
st.set_page_config(page_title="CPA Prediction App", page_icon="🔎")
st.title("CPA Prediction App 🔎")

# Sidebar menu
menu = st.sidebar.selectbox("Select Menu", ["History", "Dataset", "Prediction"])

if menu == "History":
    st.subheader("History")
    st.write("Displaying line charts for each column in the dataset.")
    for column in df_X.columns:
        plt.figure()
        plt.plot(df_ori['Date'], df_ori[column])
        plt.title(column)
        plt.xlabel('Date')
        plt.ylabel(column)
        st.pyplot(plt)

elif menu == "Dataset":
    st.subheader("Dataset")
    st.write(df_ori)

elif menu == "Prediction":
    st.subheader("Prediction")
    st.write("""
    Enter the Cost, CPC (Destination), CPM, Impression, Clicks (Destination), CTR (Destination), Conversions, and CPA at Day 1 until Day 4 (Don't forget to recheck again before click the button!):
    """)
    
    new_name_inputs = []
    with st.form("cpa_form"):
        for i in range(16):
            day = (i // 4) + 1
            metric = ["Cost", "CPC (Destination)", "CPM", "CTR (Destination)"][i % 4]
            new_name_input = st.text_input(label=f'{metric} at Day {day}:', key=f'input_{i+16}')
            new_name_inputs.append(new_name_input)
        
        if st.form_submit_button("Predict The CPA!"):
            # Get the input values
            new_name = np.array([float(new_name_input) for new_name_input in new_name_inputs]).reshape(-1, X_test.shape[1])

            # Scale the input features
            scaler = StandardScaler().fit(X_train_no_nan)
            X_train_scaled = scaler.transform(X_train_no_nan)
            X_test_scaled = scaler.transform(new_name)

            # Define the hyperparameter distribution
            param_dist = {
                'n_estimators': [10, 50, 100, 200, 500],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10, 20, 30],
                'min_samples_leaf': [1, 2, 4, 8, 16]
            }

            # Initialize the Random Forest Regressor model
            model = RandomForestRegressor(random_state=42)

            # Perform hyperparameter tuning using RandomizedSearchCV
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=5, scoring='neg_mean_squared_error', verbose=0, n_iter=20, random_state=42)
            random_search.fit(X_train_scaled, y_train_no_nan)

            # Extract the best model and fit it to the training data
            best_model = random_search.best_estimator_
            best_model.fit(X_train_scaled, y_train_no_nan)

            # Make predictions on the test data
            y_pred = best_model.predict(X_test_scaled)
            y_pred = np.round(y_pred, 0)

            # Display the predictions below the button
            st.write("Tomorrow's CPA Prediction:")
            st.write(y_pred)

st.write("""
Please refresh the website if you want to input new values.
""")

st.caption('Copyright (c) PT Ebliethos Indonesia 2024')
