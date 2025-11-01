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
print(df)
print(df.corr(numeric_only=True))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
