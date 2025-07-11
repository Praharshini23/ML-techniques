import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset from sklearn
iris = load_iris()

# Create a pandas DataFrame from the data
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the species (target) column
df['species'] = iris.target
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# Show the first 5 rows of the dataset
print(df.head())
# Show basic stats like mean, std, min, max, and quartiles
print(df.describe())
print("Mean of sepal length:", df['sepal length (cm)'].mean())
print("Median of sepal length:", df['sepal length (cm)'].median())
print("Variance of sepal length:", df['sepal length (cm)'].var())
print("Standard deviation of sepal length:", df['sepal length (cm)'].std())
print("25% Quantile of sepal length:", df['sepal length (cm)'].quantile(0.25))
print("50% Quantile (Median) of sepal length:", df['sepal length (cm)'].quantile(0.5))
print("75% Quantile of sepal length:", df['sepal length (cm)'].quantile(0.75))


# Histogram of Sepal Length
df['sepal length (cm)'].hist(color='skyblue')
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Count")
plt.show()

# Boxplot by species
sns.boxplot(x='species', y='sepal length (cm)', data=df)
plt.title("Sepal Length by Species")
plt.show()
