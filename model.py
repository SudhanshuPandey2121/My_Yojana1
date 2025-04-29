# import pandas as pd
# import numpy as np
# import re
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import LabelEncoder

# # Load the dataset (replace with your actual dataset path)
# df = pd.read_csv('Dataset/government_schemes_dataset.csv')

# # Predefined genres and keywords for classification
# genres = {
#     'Healthcare': ['health', 'medical', 'hospital', 'ayushman'],
#     'Education': ['education', 'school', 'learning', 'beti bachao', 'skill'],
#     'Employment': ['job', 'employment', 'work', 'mgnrega', 'kaushal'],
#     'Social Security': ['pension', 'assistance', 'social', 'welfare'],
#     'Housing': ['housing', 'home', 'shelter', 'awas'],
#     'Digital Empowerment': ['digital', 'technology', 'internet', 'computer'],
#     'Financial Inclusion': ['bank', 'finance', 'loan', 'jan dhan', 'mudra'],
#     'Women Empowerment': ['women', 'girl', 'matru', 'mahila'],
#     'Rural Development': ['rural', 'village', 'gram', 'panchayat'],
#     'Urban Development': ['urban', 'city', 'municipal', 'smart city'],
#     'Agriculture': ['farm', 'crop', 'agriculture', 'kisan'],
#     'Environment': ['environment', 'climate', 'pollution', 'green'],
#     'Entrepreneurship': ['entrepreneur', 'startup', 'business', 'stand up india'],
#     'Sanitation': ['sanitation', 'toilet', 'hygiene', 'swachh']
# }

# # Function to determine the genre based on scheme name
# def determine_genre(scheme_name):
#     scheme_name = scheme_name.lower()
#     for genre, keywords in genres.items():
#         if any(keyword in scheme_name for keyword in keywords):
#             return genre
#     return "General Welfare"

# # Apply genre classification to the dataset
# df['Genre'] = df['Scheme Name'].apply(determine_genre)

# # Function to generate a description based on the scheme name and genre
# def generate_description(scheme_name, genre):
#     words = re.findall(r'\w+', scheme_name.lower())
#     if 'pradhan' in words and 'mantri' in words:
#         scheme_type = "flagship"
#     else:
#         scheme_type = "government"
    
#     templates = [
#         f"A {scheme_type} scheme in the {genre} sector, aimed at improving the lives of citizens through targeted interventions.",
#         f"This {genre} initiative focuses on enhancing the welfare of the population through various {scheme_type} measures.",
#         f"A comprehensive {scheme_type} program designed to address key issues in the {genre} domain and promote overall development.",
#         f"An innovative approach to tackling challenges in the {genre} sector, this {scheme_type} scheme aims to bring about positive change.",
#         f"Targeting the {genre} aspect of societal development, this {scheme_type} initiative strives to create a meaningful impact."
#     ]
    
#     return np.random.choice(templates)

# # Generate descriptions for each scheme in the dataset
# df['Description'] = df.apply(lambda row: generate_description(row['Scheme Name'], row['Genre']), axis=1)

# # Function to get eligibility criteria
# def get_eligibility_criteria(scheme_row):
#     age = scheme_row['Applicable Age']
#     gender = scheme_row['Gender']
#     income_range = scheme_row['Income Range']
    
#     eligibility = f"Applicable Age: {age}, Gender: {gender}, Income Range: {income_range}"
#     return eligibility

# # Create a text classifier to match input scheme names with known schemes
# tfidf = TfidfVectorizer(stop_words='english')
# model = make_pipeline(tfidf, MultinomialNB())
# model.fit(df['Scheme Name'], df['Scheme Name'])  # We are training it to match scheme names

# # Save the model and DataFrame to a pickle file
# with open('government_scheme_model.pkl', 'wb') as f:
#     pickle.dump((model, df), f)

# print("Text classification model and DataFrame saved to 'government_scheme_model.pkl'")

# # Function to extract age range
# def extract_age_range(age_range):
#     age_range = re.findall(r'\d+', age_range)
#     if len(age_range) == 2:
#         return int(age_range[0]), int(age_range[1])
#     elif len(age_range) == 1:
#         return int(age_range[0]), 100  # Assume upper age limit is 100 if not provided
#     else:
#         return 0, 100  # Default age range for missing values

# # Apply age extraction
# df['Min_Age'], df['Max_Age'] = zip(*df['Applicable Age'].apply(extract_age_range))

# # Handle income range and convert to numeric values
# # Replace 'No Income Limit' with a large value (e.g., 10 billion) instead of infinity
# df['Income_Limit'] = df['Income Range'].apply(lambda x: 1e10 if x == 'No Income Limit' else int(re.findall(r'\d+', x)[0]) * 1e5)

# # Label encoding for Gender and State
# gender_encoder = LabelEncoder()
# df['Gender_Encoded'] = gender_encoder.fit_transform(df['Gender'])

# state_encoder = LabelEncoder()
# df['State_Encoded'] = state_encoder.fit_transform(df['State'])

# # Drop unneeded columns
# df_cleaned = df.drop(columns=['Scheme Name', 'Applicable Age', 'Gender', 'State', 'Income Range', 'Description'])

# # Prepare the input and output data
# X = df_cleaned[['Min_Age', 'Max_Age', 'Gender_Encoded', 'State_Encoded', 'Income_Limit']]
# y = df['Scheme Name']

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a decision tree classifier
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)

# # Save the model and encoders using pickle
# with open('decision_tree_model.pkl', 'wb') as model_file:
#     pickle.dump(clf, model_file)

# with open('gender_encoder.pkl', 'wb') as gender_file:
#     pickle.dump(gender_encoder, gender_file)

# with open('state_encoder.pkl', 'wb') as state_file:
#     pickle.dump(state_encoder, state_file)

# print("Decision tree model and encoders saved successfully!")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import re
import pickle

# Load the dataset
df = pd.read_csv('Dataset/government_schemes_dataset.csv')

# Predefined genres and keywords
genres = {
    'Healthcare': ['health', 'medical', 'hospital', 'ayushman'],
    'Education': ['education', 'school', 'learning', 'beti bachao', 'skill'],
    'Employment': ['job', 'employment', 'work', 'mgnrega', 'kaushal'],
    'Social Security': ['pension', 'assistance', 'social', 'welfare'],
    'Housing': ['housing', 'home', 'shelter', 'awas'],
    'Digital Empowerment': ['digital', 'technology', 'internet', 'computer'],
    'Financial Inclusion': ['bank', 'finance', 'loan', 'jan dhan', 'mudra'],
    'Women Empowerment': ['women', 'girl', 'matru', 'mahila'],
    'Rural Development': ['rural', 'village', 'gram', 'panchayat'],
    'Urban Development': ['urban', 'city', 'municipal', 'smart city'],
    'Agriculture': ['farm', 'crop', 'agriculture', 'kisan'],
    'Environment': ['environment', 'climate', 'pollution', 'green'],
    'Entrepreneurship': ['entrepreneur', 'startup', 'business', 'stand up india'],
    'Sanitation': ['sanitation', 'toilet', 'hygiene', 'swachh']
}

# Function to determine genre based on scheme name
def determine_genre(scheme_name):
    scheme_name = scheme_name.lower()
    for genre, keywords in genres.items():
        if any(keyword in scheme_name for keyword in keywords):
            return genre
    return "General Welfare"

# Apply initial genre
df['Genre'] = df['Scheme Name'].apply(determine_genre)

# Create TF-IDF vectorizer and genre classifier
tfidf = TfidfVectorizer(stop_words='english')
genre_clf = make_pipeline(tfidf, MultinomialNB())

# Train genre classifier
genre_clf.fit(df['Scheme Name'], df['Genre'])

# Function to predict genre using the trained classifier
def predict_genre(scheme_name):
    return genre_clf.predict([scheme_name])[0]

# Apply predicted genre
df['Predicted Genre'] = df['Scheme Name'].apply(predict_genre)

# Function to generate description based on scheme name and genre
def generate_description(scheme_name, genre):
    words = re.findall(r'\w+', scheme_name.lower())
    if 'pradhan' in words and 'mantri' in words:
        scheme_type = "flagship"
    else:
        scheme_type = "government"
    
    templates = [
        f"A {scheme_type} scheme in the {genre} sector, aimed at improving the lives of citizens through targeted interventions.",
        f"This {genre} initiative focuses on enhancing the welfare of the population through various {scheme_type} measures.",
        f"A comprehensive {scheme_type} program designed to address key issues in the {genre} domain and promote overall development.",
        f"An innovative approach to tackling challenges in the {genre} sector, this {scheme_type} scheme aims to bring about positive change.",
        f"Targeting the {genre} aspect of societal development, this {scheme_type} initiative strives to create a meaningful impact."
    ]
    
    return np.random.choice(templates)

# Apply generated descriptions
df['Generated Description'] = df.apply(lambda row: generate_description(row['Scheme Name'], row['Predicted Genre']), axis=1)

# Prepare data for ML model
le_gender = LabelEncoder()
le_state = LabelEncoder()
df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
df['State_encoded'] = le_state.fit_transform(df['State'])
df['Income_encoded'] = df['Income Range'].apply(lambda x: int(x.split()[2].replace(',', '')) if x != 'No Income Limit' else 1000)

def preprocess_age(age):
    if '-' in age:
        return int(age.split('-')[0])
    elif age == 'All Ages':
        return 0
    else:
        return int(age)

X = df[['Applicable Age', 'Gender_encoded', 'State_encoded', 'Income_encoded']].copy()
X['Applicable Age'] = X['Applicable Age'].apply(preprocess_age)
y = df['Scheme Name']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

def get_scheme_info(scheme_name):
    scheme = df[df['Scheme Name'] == scheme_name].iloc[0]
    return {
        'name': scheme_name,
        'description': scheme['Generated Description'],
        'genre': scheme['Predicted Genre']
    }

def check_eligibility(age, gender, state, income):
    eligible_schemes = []

    for _, scheme in df[df['State'] == state].iterrows():  # Filter by state
        age_range = scheme['Applicable Age'].split('-')
        
        # Check age eligibility
        if len(age_range) == 2:
            min_age, max_age = map(int, age_range)
            if not (min_age <= age <= max_age):
                continue
        elif scheme['Applicable Age'] != 'All Ages' and int(scheme['Applicable Age']) != age:
            continue

        # Check gender eligibility
        if scheme['Gender'] != 'Both' and scheme['Gender'] != gender:
            continue

        # Check state eligibility
        if scheme['State'] != state:
            continue

        # Check income eligibility
        if scheme['Income Range'] != 'No Income Limit':
            max_income = int(scheme['Income Range'].split()[2].replace(',', '')) * 100000
            if income > max_income:
                continue

        eligible_schemes.append(scheme['Scheme Name'])

    return eligible_schemes

def get_ml_recommendations(age, gender, state, income):
    try:
        gender_encoded = le_gender.transform([gender])[0]
    except ValueError:
        gender_encoded = -1  # Use a default value for unseen labels

    try:
        state_encoded = le_state.transform([state])[0]
    except ValueError:
        state_encoded = -1  # Use a default value for unseen labels

    income_encoded = income // 100000  # Convert to lakhs

    input_data = [[age, gender_encoded, state_encoded, income_encoded]]
    probabilities = clf.predict_proba(input_data)[0]
    
    # Get top 5 recommendations
    top_indices = probabilities.argsort()[-5:][::-1]
    return [clf.classes_[i] for i in top_indices]

def generate_eligibility_graph(eligible_count, total_count, state):
    labels = ['Eligible Schemes', 'Other Schemes in State']
    sizes = [eligible_count, total_count - eligible_count]
    colors = ['#4CAF50', '#FFC107']
    
    plt.figure(figsize=(8, 5))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title(f"Eligibility Overview for {state}: {eligible_count}/{total_count} Schemes")
    plt.axis('equal')
    plt.show()

# Save the model, label encoders, and other objects to a pickle file
with open('eligibility_model.pkl', 'wb') as f:
    pickle.dump({
        'model': clf,                # RandomForestClassifier model
        'genre_clf': genre_clf,      # Naive Bayes classifier for scheme genres
        'le_gender': le_gender,      # LabelEncoder for gender
        'le_state': le_state,        # LabelEncoder for state
        'df': df                     # DataFrame with schemes and details
    }, f)

print("Model and necessary objects have been saved to 'eligibility_model.pkl'.")