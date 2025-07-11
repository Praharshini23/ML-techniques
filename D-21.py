import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# 🧠 We’ll predict petal length using sepal length
X = df[['sepal length (cm)']]  # input feature
y = df['petal length (cm)']    # target

# 📚 Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📈 1. Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 📏 Evaluate Linear Regression
print("🔹 Linear Regression")
print("  Coefficients:", lr_model.coef_)
print("  Intercept:", lr_model.intercept_)
print("  R^2 Score:", r2_score(y_test, y_pred_lr))
print("  MSE:", mean_squared_error(y_test, y_pred_lr))

# 📈 2. Ridge Regression (regularized version)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# 📏 Evaluate Ridge Regression
print("\n🔸 Ridge Regression")
print("  Coefficients:", ridge_model.coef_)
print("  Intercept:", ridge_model.intercept_)
print("  R^2 Score:", r2_score(y_test, y_pred_ridge))
print("  MSE:", mean_squared_error(y_test, y_pred_ridge))

# 📊 Plot the results
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, y_pred_lr, color='blue', label='Linear Regression')
plt.plot(X_test, y_pred_ridge, color='red', linestyle='--', label='Ridge Regression')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Linear vs Ridge Regression")
plt.legend()
plt.show()
