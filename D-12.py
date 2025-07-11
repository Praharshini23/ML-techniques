# 📦 Load Libraries and Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris Dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# 🔍 Check for Missing Values
print("Missing values in each column:")
print(df.isnull().sum())

# 📊 Correlation Matrix
print("\nFeature Correlation Matrix:")
print(df.corr(numeric_only=True))

# 🔥 Heatmap of Correlations
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation between features")
plt.show()

# 📈 Pairplot
sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# 📦 Boxplots for Each Feature
for column in df.columns[:-1]:  # Skip species column
    sns.boxplot(x='species', y=column, data=df)
    plt.title(f"{column} by Species")
    plt.show()
