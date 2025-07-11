import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 🐛 Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 🎯 Predict petal length using all other features
X = df.drop(columns=['petal length (cm)'])  # Input features (3 columns)
y = df['petal length (cm)']                 # Target

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📈 Train Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 📉 Evaluate Linear Regression
print("🔹 Linear Regression:")
print("  Coefficients:", lr_model.coef_)
print("  Intercept:", lr_model.intercept_)
print("  R² Score:", r2_score(y_test, y_pred_lr))
print("  MSE:", mean_squared_error(y_test, y_pred_lr))

# 📈 Train Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# 📉 Evaluate Ridge Regression
print("\n🔸 Ridge Regression:")
print("  Coefficients:", ridge_model.coef_)
print("  Intercept:", ridge_model.intercept_)
print("  R² Score:", r2_score(y_test, y_pred_ridge))
print("  MSE:", mean_squared_error(y_test, y_pred_ridge))

# 📊 Visualization (Predicted vs Actual)
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_lr, color='blue', label='Linear Predictions')
plt.scatter(y_test, y_pred_ridge, color='red', marker='x', label='Ridge Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='black', linestyle='--')
plt.xlabel("Actual Petal Length")
plt.ylabel("Predicted Petal Length")
plt.title("Prediction Accuracy: Linear vs Ridge")
plt.legend()
plt.grid(True)
plt.show()
