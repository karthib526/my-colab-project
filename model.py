import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('data/winequality-red.csv', sep=';') 
 
#print(df.head())
 


#print(df.info())

#print(df.describe())
df.corr(numeric_only=True)

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
target = 'quality'  # change this to your actual target column

# Compute correlation of each feature with target
corr_with_target = df.corr()[target].sort_values(ascending=False)

# Display all correlations
print(corr_with_target)
corr_with_target.drop(target).plot(kind='bar', figsize=(10,5))
plt.title('Correlation of Features with Target')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.show()

