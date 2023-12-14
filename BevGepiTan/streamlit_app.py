import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('BevGepiTan/datos_merged_1986_2023.csv')

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

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_linear = linear_model.predict(X_test_scaled)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# XGBoost Regression
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=7, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Decision Tree Regression
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

# Neural Network (MLPRegressor)
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn_model.fit(X_train_scaled, y_train)
y_pred_nn = nn_model.predict(X_test_scaled)

# Streamlit app
st.title('Energy Prediction App')
st.write('Performing energy prediction using different regression models')

# Select model
selected_model = st.selectbox('Select Model', ['Linear Regression', 'Polynomial Regression', 'XGBoost Regression', 'Decision Tree Regression', 'Random Forest Regression', 'Neural Network'])

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

elif selected_model == 'Decision Tree Regression':
    st.header('Decision Tree Regression Metrics')
    st.write("R2 Score:", r2_score(y_test, y_pred_dt))
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred_dt))

elif selected_model == 'Random Forest Regression':
    st.header('Random Forest Regression Metrics')
    st.write("R2 Score:", r2_score(y_test, y_pred_rf))
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))

elif selected_model == 'Neural Network':
    st.header('Neural Network Metrics')
    st.write("R2 Score:", r2_score(y_test, y_pred_nn))
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred_nn))
