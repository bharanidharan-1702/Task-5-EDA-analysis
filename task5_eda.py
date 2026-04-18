# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# LOAD DATA
# ===============================
print("🚀 Starting EDA...")

df = pd.read_csv('../train.csv')

# ===============================
# BASIC INFO
# ===============================
print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Description:")
print(df.describe())

print("\n🎯 Survival Count:")
print(df['Survived'].value_counts())

# ===============================
# DATA CLEANING
# ===============================
df['Age'].fillna(df['Age'].mean(), inplace=True)

# ===============================
# VISUALIZATIONS
# ===============================

# 1. Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# ------------------------------

# 2. Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# ------------------------------

# 3. Age Distribution
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")
plt.show()

# ------------------------------

# 4. Age vs Survival
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

# ------------------------------

# 5. Fare Distribution
sns.histplot(df['Fare'], kde=True)
plt.title("Fare Distribution")
plt.show()

# ------------------------------

# 6. Survival by Class
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Class")
plt.show()

# ------------------------------

# 7. Correlation Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------

# 8. Pairplot
sns.pairplot(df[['Age','Fare','Survived']])
plt.show()

# ===============================
# FINISHED
# ===============================
print("\n✅ EDA Completed Successfully!")