import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("Dataset/government_schemes_dataset.csv")

# Function to extract age range
def extract_age_range(age_range):
    age_range = re.findall(r'\d+', age_range)
    if len(age_range) == 2:
        return int(age_range[0]), int(age_range[1])
    elif len(age_range) == 1:
        return int(age_range[0]), 100  # Assume upper age limit is 100 if not provided
    else:
        return 0, 100  # Default age range for missing values

# Apply age extraction
df['Min_Age'], df['Max_Age'] = zip(*df['Applicable Age'].apply(extract_age_range))

# Handle income range and convert to numeric values
df['Income_Limit'] = df['Income Range'].apply(lambda x: 1e10 if x == 'No Income Limit' else int(re.findall(r'\d+', x)[0]) * 1e5)

# Label encoding for Gender and State
gender_encoder = LabelEncoder()
df['Gender_Encoded'] = gender_encoder.fit_transform(df['Gender'])

state_encoder = LabelEncoder()
df['State_Encoded'] = state_encoder.fit_transform(df['State'])

# Drop unneeded columns
df_cleaned = df.drop(columns=['Scheme Name', 'Applicable Age', 'Gender', 'State', 'Income Range'])

# Prepare the input and output data
X = df_cleaned[['Min_Age', 'Max_Age', 'Gender_Encoded', 'State_Encoded', 'Income_Limit']]
y = df['Scheme Name']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Save the model and encoders using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

with open('gender_encoder.pkl', 'wb') as gender_file:
    pickle.dump(gender_encoder, gender_file)

with open('state_encoder.pkl', 'wb') as state_file:
    pickle.dump(state_encoder, state_file)

print("Model and encoders saved successfully!")
