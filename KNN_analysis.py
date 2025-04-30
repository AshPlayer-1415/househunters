# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Step 2: Load the cleaned dataset
df = pd.read_csv('data/Home_rentals_Data.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Clean currency columns
def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace('$', '').replace(',', ''))
    return x

df['RENT_PER_MONTH'] = df['RENT_PER_MONTH'].apply(clean_currency)
df['SQUARE_FEET'] = df['SQUARE_FEET'].apply(lambda x: float(str(x).replace(',', '')))
df['RENT_PRICE_PER_SQFT'] = df['RENT_PRICE_PER_SQFT'].apply(clean_currency)
df['RENT_TOTAL_COST_AFTER_30_YEARS'] = df['RENT_TOTAL_COST_AFTER_30_YEARS'].apply(clean_currency)

# Step 3: Select input features and target
features = ['BEDS', 'BATHS', 'SQUARE_FEET', 'LATITUDE', 'LONGITUDE']
target = 'RENT_PER_MONTH'

X = df[features]
y = df[target]

# Step 4: Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train KNN model
k = 5
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X_train_scaled, y_train)

# Step 7: Predict on test set
y_pred = knn_model.predict(X_test_scaled)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Step 9: Display actual vs predicted with features
comparison = X_test.copy()
comparison['Actual Rent'] = y_test.values
comparison['Predicted Rent'] = y_pred
print("\nSample Predictions with Features:")
print(comparison.head())

# Step 10: Predict rent for Adelanto property
adelanto_home = pd.DataFrame({
    'BEDS': [4],
    'BATHS': [1],
    'SQUARE_FEET': [972],
    'LATITUDE': [34.58544],
    'LONGITUDE': [-117.40139]
})

adelanto_scaled = scaler.transform(adelanto_home)
adelanto_predicted_rent = knn_model.predict(adelanto_scaled)

print(f"\nPredicted Rent for 4-bed, 1-bath, 972 sqft in Adelanto, CA: ${adelanto_predicted_rent[0]:.2f}")
