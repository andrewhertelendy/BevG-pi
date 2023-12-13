import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# Load the dataset
data = pd.read_excel('datos_merged_1986_2023.xlsx')

# Data preprocessing
columns_to_keep = ['popularity', 'danceability', 'year', 'valence', 'speechiness', 'loudness', 'energy', 'principal_artist_followers', 'album_total_tracks', 'acousticness']
columns_to_drop = [col for col in data.columns if col not in columns_to_keep]
data_processed = data.drop(columns=columns_to_drop, axis=1)
data_processed.dropna(inplace=True)

# Separating features and target
X = data_processed.drop('energy', axis=1)
y = data_processed['energy']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# XGBoost Regression
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=7, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Streamlit app
st.title('Energy Prediction App')
st.write('Performing energy prediction using different regression models')

# Select model
selected_model = st.selectbox('Select Model', ['Linear Regression', 'Polynomial Regression', 'XGBoost Regression'])

# Display metrics based on selected model
if selected_model == 'Linear Regression':
    st.header('Linear Regression Metrics')
    st.write("R2 Score:", r2_score(y_test, y_pred_linear))
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred_linear))

elif selected_model == 'Polynomial Regression':
    st.header('Polynomial Regression Metrics')
    st.write("R2 Score:", r2_score(y_test, y_pred_poly))
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred_poly))

elif selected_model == 'XGBoost Regression':
    st.header('XGBoost Regression Metrics')
    st.write("R2 Score:", r2_score(y_test, y_pred_xgb))
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred_xgb))
