import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
data = pd.read_excel('datos_merged_1986_2023.xlsx')

# Define the Streamlit app
def main():
    st.title('Energy Prediction App')
    st.write('Performing energy prediction using different regression models')

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
    st.header('Linear Regression')
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    st.write("R2 Score (Linear Regression):", r2_score(y_test, y_pred_linear))
    st.write("Mean Squared Error (Linear Regression):", mean_squared_error(y_test, y_pred_linear))

    # Polynomial Regression
    st.header('Polynomial Regression')
    poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    st.write("R2 Score (Polynomial Regression):", r2_score(y_test, y_pred_poly))
    st.write("Mean Squared Error (Polynomial Regression):", mean_squared_error(y_test, y_pred_poly))

# Run the Streamlit app
if __name__ == '__main__':
    main()
