import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Load data
data=pd.read_csv("/mnt/c/Users/rhwns/OneDrive/바탕 화면/lab/project1/WorldHappinessReport2024.csv")

# drop all missing value in social sport
data = data.dropna(subset=['Social support'])


#impute missing values with prediction model
for column in data.columns:
    if data[column].isnull().any() and column not in ['Life Ladder', 'Social support Category']:
        # Define predictors and target
        X = data[['Life Ladder', 'Social support']]
        y = data[column]
        
        # Train on non-missing data
        X_train = X.loc[~y.isna()]
        y_train = y.dropna()
        
        if len(X_train) > 0:
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict missing values
            X_missing = X.loc[y.isna()]
            data.loc[y.isna(), column] = model.predict(X_missing)

# Define bins based on min and max values
bins = [0, 0.3, 0.4, 0.6, 0.7, 1]
labels = ['Very Low', 'Low', 'Moderate', 'High','Extreme']

# Create binned feature
data['Social support Category'] = pd.cut(data['Social support'], bins=bins, labels=labels)

# Create Happiness Index
data['Happiness Index'] = (data['Positive affect'] - data['Negative affect']) / (data['Positive affect'] + data['Negative affect'] + 1)

# Define thresholds for categorization
def categorize_happiness(index):
    if index < -0.2:
        return 'Very Sad'
    elif -0.2 <= index < -0.1:
        return 'Sad'
    elif -0.1 <= index < 0.1:
        return 'Neutral'
    elif 0.1 <= index < 0.3:
        return 'Happy'
    else:
        return 'Very Happy'

# Apply categorization
data['Happiness Category'] = data['Happiness Index'].apply(categorize_happiness)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

data['Social support Category'] = label_encoder.fit_transform(data['Social support Category'])

# Define the mapping dictionary for Happiness Category
happiness_mapping = {
    'Very Sad': 0,
    'Sad': 1,
    'Neutral': 2,
    'Happy': 3,
    'Very Happy': 4
}
data['Happiness Category'] = data['Happiness Category'].map(happiness_mapping)

data= data.drop(columns=['Country name']) # drop country name

# Train-valid-test split (0.7, 0.2, 0.1)
X = data.drop(columns=['Happiness Index', 'Happiness Category'])
y = data['Happiness Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

X_train.to_csv('/mnt/c/Users/rhwns/OneDrive/바탕 화면/lab/project1/data/X_train.csv', index=False)
X_valid.to_csv('/mnt/c/Users/rhwns/OneDrive/바탕 화면/lab/project1//data/X_valid.csv', index=False)
X_test.to_csv('/mnt/c/Users/rhwns/OneDrive/바탕 화면/lab/project1/data/X_test.csv', index=False)
y_train.to_csv('/mnt/c/Users/rhwns/OneDrive/바탕 화면/lab/project1/data/y_train.csv', index=False)
y_valid.to_csv('/mnt/c/Users/rhwns/OneDrive/바탕 화면/lab/project1/data/y_valid.csv', index=False)
y_test.to_csv('/mnt/c/Users/rhwns/OneDrive/바탕 화면/lab/project1/data/y_test.csv', index=False)