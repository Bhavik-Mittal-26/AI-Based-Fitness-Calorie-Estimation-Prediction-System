import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv('calories.csv')

# 2. Feature Engineering
df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
df['Intensity_Score'] = df['Heart_Rate'] * df['Duration']
df['Metabolic_Stress'] = df['Body_Temp'] * df['Heart_Rate']

# 3. Label Encoding
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# 4. Train/Test Split
X = df.drop(['User_ID', 'Calories'], axis=1)
y = df['Calories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Hyperparameter Tuning (using simpler grid for fast execution)
param_grid = {
    'n_estimators': [200],
    'max_depth': [5],
    'learning_rate': [0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

random_search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42), 
    param_distributions=param_grid, 
    n_iter=1, 
    scoring='neg_mean_squared_error', 
    cv=3, 
    random_state=42, 
    n_jobs=-1
)
random_search.fit(X_train_scaled, y_train)
best_xgb = random_search.best_estimator_

# 7. Save model, scaler, and label encoder
joblib.dump(best_xgb, 'best_xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("All assets (model, scaler, label_encoder) saved successfully!")
