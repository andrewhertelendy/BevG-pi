# %%
import pandas as pd
import streamlit as st
# from google.colab import drive
# drive.mount('/content/gdrive')
# data = pd.read_excel('/content/gdrive/MyDrive/SpotifyRegression/datos_merged_1986_2023.xlsx')

data = pd.read_excel('datos_merged_1986_2023.xlsx')


# %%
# Columns to keep
columns_to_keep = ['popularity', 'danceability', 'year', 'valence', 'speechiness', 'loudness', 'energy', 'principal_artist_followers', 'album_total_tracks', 'acousticness']

# Drop columns not present in the columns_to_keep list
columns_to_drop = [col for col in data.columns if col not in columns_to_keep]
data = data.drop(columns=columns_to_drop, axis=1)



# %%
from sklearn.model_selection import train_test_split, GridSearchCV
data.dropna(inplace=True)  # Dropping rows with missing values for simplicity

# Separate features (X) and target variable (y)
X = data.drop('energy', axis=1)  # Features
y = data['energy']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler

# Assuming 'X_train', 'X_test', 'y_train', 'y_test' are already defined from previous steps



# Creating a Linear Regression model
model = LinearRegression()

# # Hyperparameter grid for GridSearchCV
# param_grid = {
#     'fit_intercept': [True, False],
#     # Other hyperparameters can be added based on the available options
# }

# # Perform GridSearchCV for hyperparameter tuning
# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')  # Change scoring as needed
# grid_search.fit(X_train, y_train)
model.fit(X_train, y_train)

# Best parameters found by GridSearchCV
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# # Evaluate the model with the best parameters on the test set
# best_model = grid_search.best_estimator_
y_pred = model.predict(X_test)

# Model Evaluation
from sklearn.metrics import r2_score, mean_squared_error
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# %%
# from xgboost import XGBRegressor
# from sklearn.model_selection import GridSearchCV

# # Define a parameter grid to search through
# param_grid = {
#     'n_estimators': [100, 500, 1000],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 5, 7]
# }

# # Creating and training the XGBoost Regressor model
# model = XGBRegressor(random_state=42)

# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2')
# grid_search.fit(X_train, y_train)

# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# # Model Evaluation
# print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))


# %%
from xgboost import XGBRegressor

# Creating and training the XGBoost Regressor model
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=7, random_state=42)
model.fit(X_train, y_train)


st.title('Danceability Predictor')
st.sidebar.header('User Input Features')
input_data = {
    'Popularity': st.sidebar.slider('Popularity', float(X['popularity'].min()), float(X['popularity'].max()), float(X['popularity'].mean())),
    'Year': st.sidebar.slider('Year', int(X['year'].min()), int(X['year'].max()), int(X['year'].mean())),
    # Add sliders for other features here
}

input_features = pd.DataFrame([input_data])

def predict_danceability(input_df):
    prediction = model.predict(input_df)
    return prediction

if st.sidebar.button('Predict'):
    predicted_danceability = predict_danceability(input_features)
    st.write('Predicted Danceability:', predicted_danceability)


# Predicting on the test set
y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))



# %%



