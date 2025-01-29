import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the dataset
file_path = r"C:\Users\Khan\OneDrive - Iqra University Islamabad Chak Shahzad\software engineering\5th Semester\Data Science\presentation\Student Depression Dataset.xlsx"
dataset = pd.read_excel(file_path)

# Display dataset information
print("Dataset Info:")
print(dataset.info())

print("\nFirst 5 Rows of Dataset:")
print(dataset.head())

# Descriptive statistics for numerical columns
print("\nDescriptive Statistics:")
print(dataset.describe())

# Unique values in categorical columns
categorical_columns = dataset.select_dtypes(include='object').columns
unique_values = {col: dataset[col].nunique() for col in categorical_columns}
print("\nUnique Values in Categorical Columns:")
print(unique_values)

# Set style for plots
sns.set(style="whitegrid")

# Plot 1: Distribution of Age
plt.figure(figsize=(8, 5))
sns.histplot(dataset['Age'], kde=True, bins=15, color="blue")
plt.title("Age Distribution", fontsize=14)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Plot 2: Depression vs. Academic Pressure
plt.figure(figsize=(8, 5))
sns.boxplot(x="Depression", y="Academic Pressure", data=dataset, palette="Set2", hue=None)
plt.title("Academic Pressure vs. Depression", fontsize=14)
plt.xlabel("Depression (0 = No, 1 = Yes)")
plt.ylabel("Academic Pressure")
plt.show()

# Plot 3: Gender Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", data=dataset, palette="pastel", hue=None)
plt.title("Gender Distribution", fontsize=14)
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Predictive Analysis: Logistic Regression
# Selecting features and target for predictive analysis
features = dataset[['Academic Pressure', 'Financial Stress', 'Study Satisfaction']]
target = dataset['Depression']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = logistic_model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_rep)