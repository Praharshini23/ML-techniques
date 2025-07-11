# Day 3: Mini Project 2 - Cross Validation Comparison
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create classifiers
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear')
}

# Store average scores
cv_scores = {}

# Perform 5-fold cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)  # 5 folds
    cv_scores[name] = scores.mean()
    print(f"{name} - Accuracy Scores: {scores}")
    print(f"{name} - Average Accuracy: {scores.mean():.4f}")

# Plot the results
plt.figure(figsize=(8, 5))
sns.barplot(x=list(cv_scores.keys()), y=list(cv_scores.values()), palette="pastel")
plt.title("Model Comparison using 5-Fold Cross-Validation")
plt.ylabel("Average Accuracy")
plt.ylim(0.9, 1.0)
plt.show()
