import streamlit as st
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.evaluate import bias_variance_decomp

# Configure the page
st.set_page_config(page_title='House Price Prediction Bias-Variance Analysis', layout='centered', initial_sidebar_state='expanded')

st.title("🏠 House Price Prediction Bias-Variance Analysis")
st.write("""
This demo showcases the bias-variance tradeoff using a **Decision Tree Regressor** on a house price prediction dataset.
""")

# Load dataset
df = pd.read_csv('Clean_HousePrice.csv')
cat_enc = pd.get_dummies(df[['furnishing status','mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']], drop_first=True).astype('int64')
df1 = df.drop(['furnishing status','mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'], axis=1)
df_final = pd.concat([df1, cat_enc], axis=1)

# Split the data
x = df_final.drop('price', axis=1)
y = df_final['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale the features
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convert to NumPy arrays
x_train_np = np.array(x_train)
y_train_np = np.array(y_train)
x_test_np = np.array(x_test)
y_test_np = np.array(y_test)

# Load the pickled model
with open('decision_tree_model.pkl', 'rb') as file:
    base_model = pickle.load(file)

# Function to perform bias-variance analysis using mlxtend
def bias_variance_analysis_mlxtend(model, x_train, y_train, x_test, y_test, param_range):
    avg_expected_loss = []
    avg_bias = []
    avg_var = []

    for param in param_range:
        model.set_params(max_depth=param)
        mean_error, bias, var = bias_variance_decomp(
            model, x_train, y_train, x_test, y_test, loss='mse', random_seed=42
        )
        avg_expected_loss.append(mean_error)
        avg_bias.append(bias)
        avg_var.append(var)
    
    return avg_expected_loss, avg_bias, avg_var

# Sidebar user inputs
st.sidebar.header('Bias-Variance Analysis Parameters')
min_depth = st.sidebar.slider('Minimum max_depth', 1, 20, 1)
max_depth = st.sidebar.slider('Maximum max_depth', 1, 20, 20)
param_step = st.sidebar.slider('Step size for max_depth', 1, 5, 1)
param_range = range(min_depth, max_depth + 1, param_step)

# Perform analysis on button click
if st.sidebar.button('Run Bias-Variance Analysis'):
    with st.spinner('Performing Bias-Variance Analysis...'):
        avg_expected_loss, avg_bias, avg_var = bias_variance_analysis_mlxtend(base_model, x_train_np, y_train_np, x_test_np, y_test_np, param_range)
        st.success('Analysis Completed!')

        # Plotting the results
        st.subheader("Bias-Variance Tradeoff Analysis")
        fig, ax = plt.subplots()
        ax.plot(param_range, avg_expected_loss, label='Expected Loss', marker='o')
        ax.plot(param_range, avg_bias, label='Bias^2', marker='o')
        ax.plot(param_range, avg_var, label='Variance', marker='o')
        ax.set_xlabel('Model Complexity (max_depth)')
        ax.set_ylabel('Error')
        ax.set_title('Bias-Variance Tradeoff')
        ax.legend()
        st.pyplot(fig)

# Display dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())
