import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load data
df = pd.read_csv("Significant_Earthquakes (1).csv")

# Drop irrelevant columns
df.drop(columns=['Unnamed: 0', 'id', 'updated', 'status'], inplace=True)

# Keep only earthquakes
df = df[~df['type'].isin([
    'nuclear explosion', 'explosion', 'rock burst',
    'mine collapse', 'volcanic eruption'
])]
df.drop(columns=['type'], inplace=True)

# Handle missing values
df["depth"] = df["depth"].fillna(df["depth"].median())
df["rms"] = df["rms"].fillna(df["rms"].mean())
df["place"] = df["place"].fillna(df["place"].mode()[0])
df["depthError"] = df["depthError"].fillna(df["depthError"].mean())
df["magNst"] = df["magNst"].fillna(df["magNst"].median())
df["gap"] = df["gap"].fillna(df["gap"].median())
df["nst"] = df["nst"].fillna(df["nst"].median())
df["dmin"] = df["dmin"].fillna(df["dmin"].median())
df["horizontalError"] = df["horizontalError"].fillna(df["horizontalError"].median())
df["magError"] = df["magError"].fillna(df["magError"].median())

# Convert time column to datetime and extract parts
df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df.drop(columns=['time'], inplace=True)

# Extract country from place and drop it
df["country"] = df["place"].str.split(", ").str[-1]
df.drop(columns=['place'], inplace=True)

# Encode 'magType' using LabelEncoder
le_magType = LabelEncoder()
df['magType'] = le_magType.fit_transform(df['magType'].astype(str))

# Drop other categorical or unnecessary columns
df.drop(columns=['magSource', 'country', 'locationSource', 'depthError', 'net'], inplace=True)

# Define feature columns and target
X_cols = ['latitude', 'longitude', 'depth', 'magType', 'nst', 'gap',
          'dmin', 'rms', 'horizontalError', 'year', 'magNst', 'magError']
y_col = 'mag'

X = df[X_cols]
y = df[y_col]

# Ensure all columns are numeric
assert X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all(), "Non-numeric values found in features!"

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Double-check for NaNs in scaled data
X_train_scaled = np.nan_to_num(X_train_scaled)
X_test_scaled = np.nan_to_num(X_test_scaled)
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

# Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=1500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


import pickle

# Save trained RandomForestRegressor model
with open('earthquake_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the label encoder
with open('magType_encoder.pkl', 'wb') as f:
    pickle.dump(le_magType, f)

print("Model, scaler, and magType encoder saved successfully using pickle.")
