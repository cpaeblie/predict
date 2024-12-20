import numpy as np
import seaborn as sns
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import math
from sklearn.metrics import mean_squared_error
from numpy import array
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.svm import SVR
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error
from numpy import array
from scipy.stats import kurtosis, skew
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[out_end_ix - 1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
def stats_features(input_data):
    inp = list()
    for i in range(len(input_data)):
        inp2=list()
        inp2=input_data[i]
        min=float(np.min(inp2))
        max=float(np.max(inp2))
        diff=(max-min)
        std=float(np.std(inp2))
        mean=float(np.mean(inp2))
        median=float(np.median(inp2))
        kurt=float(kurtosis(inp2))
        sk=float(skew(inp2))
        inp2=np.append(inp2,min)
        inp2=np.append(inp2,max)
        inp2=np.append(inp2,diff)
        inp2=np.append(inp2,std)
        inp2=np.append(inp2,mean)
        inp2=np.append(inp2,median)
        inp2=np.append(inp2,kurt)
        inp2=np.append(inp2,sk)
        #print(list(inp2))
        inp=np.append(inp,inp2)
    inp=inp.reshape(len(input_data),-1)
    #print(inp)
    return inp
import pandas as pd
zymuno_df = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/main/ad%20final.csv', delimiter=',')
ads_1 = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/refs/heads/main/ads_1.csv', delimiter=',')
ads_2 = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/refs/heads/main/ads_2.csv', delimiter=',')
ads_4 = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/refs/heads/main/ads_4.csv', delimiter=',')
ads_5 = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/refs/heads/main/ads_5.csv', delimiter=',')
ads_6 = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/refs/heads/main/ads_6.csv', delimiter=',')
ads_7 = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/refs/heads/main/ads_7.csv', delimiter=',')
ads_8 = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/refs/heads/main/ads_8.csv', delimiter=',')
ads_9 = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/refs/heads/main/ads_9.csv', delimiter=',')
ads_10 = pd.read_csv('https://raw.githubusercontent.com/cpaeblie/predik/refs/heads/main/ads_10.csv', delimiter=',')
df_ori = zymuno_df
df_ori['Date'] = pd.to_datetime(df_ori['Date'])
df_X = df_ori[['Cost','CPC (Destination)','CPM','CTR (Destination)','CPA']]
in_seq = df_X.astype(float).values
#out_seq = df_y.astype(float).values

#in_seq1 = in_seq.reshape(in_seq.shape[0], in_seq.shape[1])
#out_seq = out_seq.reshape((len(out_seq), 1))

#from numpy import hstack
#dataset = hstack((in_seq1, out_seq))


n_steps_in, n_steps_out = 5, 1
X, y = split_sequences(in_seq, n_steps_in, n_steps_out)

n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = False)
X_train = stats_features(X_train)
X_test = stats_features(X_test)


df_new=df_ori[['Date','CPA']]
df_new.set_index('Date')

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
X_train_no_nan = X_train[~np.isnan(X_train).any(axis=1)]
X_test_no_nan = X_test[~np.isnan(X_test).any(axis=1)]

y_train_no_nan = y_train[~np.isnan(y_train)]
y_test_no_nan = y_test[~np.isnan(y_test)]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_no_nan = X_train[~np.isnan(X_train).any(axis=1)]
X_test_no_nan = X_test[~np.isnan(X_test).any(axis=1)]

y_train_no_nan = y_train[~np.isnan(y_train)]
y_test_no_nan = y_test[~np.isnan(y_test)]
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
# Create the title and description
st.set_page_config(page_title="CPA Prediction App", page_icon="🔎", layout="wide")
st.title("CPA Prediction App 🔎")
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Select a page:", ["History", "Dataset", "Prediction"])

if menu == "History":
    # History Page
    st.title("History")
    st.write("This section displays line charts of each column in the dataset, providing insights into trends over time.")

    # Dropdown to select dataset
    dataset_options = {
        "First Ad": ads_1,
        "Second Ad": ads_2,
	"Third Ad": zymuno_df,
        "Fourth Ad": ads_4,
        "Fifth Ad": ads_5,
        "Sixth Ad": ads_6,
        "Seventh Ad": ads_7,
        "Eighth Ad": ads_8,
        "Ninth Ad": ads_9,
        "Tenth Ad": ads_10
    }

    selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()))

    # Filtered DataFrame based on selection
    df_filtered = dataset_options[selected_dataset]

    # Ensure 'Date' column is in datetime format
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])

    # Apply filter based on the selected dataset's specific day limit
    day_limits = {
        "First Ad": 123,
        "Second Ad": 110,
	"Third Ad": 71,
        "Fourth Ad": 75,
        "Fifth Ad": 72,
        "Sixth Ad": 70,
        "Seventh Ad": 61,
        "Eighth Ad": 60,
        "Ninth Ad": 47,
        "Tenth Ad": 49
    }

    # Get the limit for the selected dataset
    limit_days = day_limits.get(selected_dataset, 70)  # Default to 70 if not found

    # Filter to limit the DataFrame based on the specified number of days
    df_filtered = df_filtered.iloc[:limit_days]

    # Date vs CPA
    st.subheader("CPA Over Time")
    st.line_chart(df_filtered.set_index('Date')['CPA'], use_container_width=True)

    # Date vs Cost
    st.subheader("Cost Over Time")
    st.line_chart(df_filtered.set_index('Date')['Cost'], use_container_width=True)

    # Date vs CPC (Destination)
    st.subheader("CPC (Destination) Over Time")
    st.line_chart(df_filtered.set_index('Date')['CPC (Destination)'], use_container_width=True)

    # Date vs CPM
    st.subheader("CPM Over Time")
    st.line_chart(df_filtered.set_index('Date')['CPM'], use_container_width=True)

    # Date vs CTR (Destination)
    st.subheader("CTR (Destination) Over Time")
    st.line_chart(df_filtered.set_index('Date')['CTR (Destination)'], use_container_width=True)
	
elif menu == "Dataset":
    # Dataset Page
    st.title("Dataset")
    st.write("This section provides an overview of the datasets used for analysis, allowing users to select and explore the data for deeper insights.")

    # Dropdown to select dataset for analysis
    dataset_options = {
        "First Ad": ads_1,
        "Second Ad": ads_2,
	"Third Ad": zymuno_df,
        "Fourth Ad": ads_4,
        "Fifth Ad": ads_5,
        "Sixth Ad": ads_6,
        "Seventh Ad": ads_7,
        "Eighth Ad": ads_8,
        "Ninth Ad": ads_9,
        "Tenth Ad": ads_10
    }

    # Dropdown to select dataset
    selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()))

    # Filtered DataFrame based on selection
    df_ori = dataset_options[selected_dataset]

    # Display the selected dataset
    st.dataframe(df_ori)

    # Correlation Analysis
    st.write("This section displays scatter plots illustrating the correlations between key features in the dataset.")

    # Define the specific pairs to analyze
    pairs = [
        ('Cost', 'CPA'),
        ('CPC (Destination)', 'CPA'),
        ('CPM', 'CPA'),
        ('CTR (Destination)', 'CPA')
    ]

    for feature1, feature2 in pairs:
        # Scatter plot with regression line
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df_ori, x=feature1, y=feature2, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        plt.title(f'Scatter Plot with Regression Line: {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

        # Show the plot in Streamlit
        st.pyplot(plt)

        # Calculate correlation
        correlation_value = df_ori[feature1].corr(df_ori[feature2])
        st.write(f"The correlation coefficient between **{feature1}** and **{feature2}** is **{correlation_value:.2f}**. This indicates a {'positive' if correlation_value > 0 else 'negative'} correlation.")

        # Additional descriptions for specific pairs
        if feature1 == 'Cost' and feature2 == 'CPA':
            st.write("This scatter plot shows the relationship between total Cost and Cost Per Acquisition (CPA). A positive correlation suggests that as total spending increases, the cost to acquire each customer may also increase.")
        elif feature1 == 'CPC (Destination)' and feature2 == 'CPA':
            st.write("This scatter plot illustrates the relationship between CPC and CPA. A positive correlation may imply that higher costs per click lead to higher costs per acquisition.")
        elif feature1 == 'CPM' and feature2 == 'CPA':
            st.write("This scatter plot shows the relationship between CPM and CPA. A positive correlation might suggest that as the cost per 1,000 impressions increases, the cost per acquisition also tends to increase.")
        elif feature1 == 'CTR (Destination)' and feature2 == 'CPA':
            st.write("This scatter plot illustrates the relationship between CTR and CPA. A positive correlation suggests that as the click-through rate increases, the cost per acquisition also increases. This may indicate that while more users are engaging with the ad, the quality of those clicks may not be leading to conversions at a lower cost, or that higher engagement is associated with higher-cost campaigns.")
	      	    
elif menu == "Prediction":
    st.title("Prediction")
    st.write("""This is a CPA Prediction App that uses machine learning algorithms to predict the Cost Per Acquisition (CPA) for a given set of input features Cost, CPC (Destination), CPM, CTR (Destination) for the 5 days before tomorrow.
""")
    st.write("""
Enter the Cost, CPC (Destination), CPM, CTR (Destination) at Day 1 until Day 5:
""")
# Create the input widgets for the new name
    new_name_inputs = []
    with st.form("cpa_form"):
        for i in range(20):
            day = (i // 4) + 1
            metric = i % 4
            if metric == 0:
                metric = "Cost"
            elif metric == 1:
                metric = "CPC (Destination)"
            elif metric == 2:
                metric = "CPM"
            else:
                metric = "CTR (Destination)"
        
            new_name_input = st.text_input(label=f'{metric} at Day {day}:', key=f'input_{i+20}')
            new_name_inputs.append(new_name_input)
        if st.form_submit_button("Predict The CPA!"):
            # Get the input values
            new_name = np.array([float(new_name_input) for new_name_input in new_name_inputs]).reshape(-1, X_test.shape[1])
            # Remaining code...

            # Scale the input features
            scaler = StandardScaler().fit(X_train_no_nan)
            X_train_scaled = scaler.transform(X_train_no_nan)
            X_test_scaled = scaler.transform(new_name)

        # Initialize the Random Forest Regressor model
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train_scaled, y_train_no_nan)

        # Make predictions on the test data
            y_pred = model.predict(X_test_scaled)
            y_pred = np.round(y_pred, 0)

        # Display the predictions in the sidebar
            st.sidebar.write("Tomorrow's CPA Prediction:")
            st.sidebar.write(y_pred)
    st.write("""
    Don't forget to recheck again before click the button
""")
    st.write("""
    Please refresh the website if you want input new values
""")

	    
st.caption('Copyright (c) PT Ebliethos Indonesia 2024')
