from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import pickle
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from scheme_data import recommend_schemes, get_scheme_by_id, get_scheme_details

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key and log if it's available
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
else:
    logger.info("GEMINI_API_KEY found")

# Initialize the Google Generative AI SDK
try:
    genai.configure(api_key=api_key)
    logger.info("Successfully configured Gemini AI")
except Exception as e:
    logger.error(f"Error configuring Gemini AI: {str(e)}")

# Set up the generative model configurations
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the model
try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction="You are trained to assist users in identifying the most suitable government scheme based on their requirements. Format the response in valid HTML with appropriate bold sections and even make the list headings in bold.",
    )
    logger.info("Successfully initialized Gemini model")
except Exception as e:
    logger.error(f"Error initializing Gemini model: {str(e)}")

app = Flask(__name__)

# Create a chat session
try:
    chat_session = model.start_chat(history=[])
    logger.info("Successfully created chat session")
except Exception as e:
    logger.error(f"Error creating chat session: {str(e)}")

# Load the models and datasets
with open('government_scheme_model.pkl', 'rb') as f:
    text_model, df = pickle.load(f)

with open('model.pkl', 'rb') as f:
    eligibility_model = pickle.load(f)

with open('gender_encoder.pkl', 'rb') as f:
    gender_encoder = pickle.load(f)

with open('state_encoder.pkl', 'rb') as f:
    state_encoder = pickle.load(f)

with open('eligibility_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

clf = model_data['model']
genre_clf = model_data['genre_clf']
le_gender = model_data['le_gender']
le_state = model_data['le_state']
df = model_data['df']

# Function to determine genre based on scheme name
def determine_genre(scheme_name):
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
    scheme_name = scheme_name.lower()
    for genre, keywords in genres.items():
        if any(keyword in scheme_name for keyword in keywords):
            return genre
    return "General Welfare"

# Function to generate description based on scheme name and genre
def generate_description(scheme_name, genre):
    words = re.findall(r'\w+', scheme_name.lower())
    scheme_type = "flagship" if 'pradhan' in words and 'mantri' in words else "government"
    
    templates = [
        f"A {scheme_type} scheme in the {genre} sector, aimed at improving the lives of citizens through targeted interventions.",
        f"This {genre} initiative focuses on enhancing the welfare of the population through various {scheme_type} measures.",
        f"A comprehensive {scheme_type} program designed to address key issues in the {genre} domain and promote overall development.",
        f"An innovative approach to tackling challenges in the {genre} sector, this {scheme_type} scheme aims to bring about positive change.",
        f"Targeting the {genre} aspect of societal development, this {scheme_type} initiative strives to create a meaningful impact."
    ]
    
    return np.random.choice(templates)

# Function to predict genre using the trained classifier
def predict_genre(scheme_name):
    return genre_clf.predict([scheme_name])[0]

# Function to get scheme information based on scheme name
def get_scheme_info(scheme_name):
    predicted_scheme = text_model.predict([scheme_name])[0]
    scheme = df[df['Scheme Name'] == predicted_scheme].iloc[0]
    
    return {
        'Scheme Name': scheme['Scheme Name'],
        'Genre': scheme['Genre'],
        'Description': scheme['Description'],
        'Eligibility Criteria': f"Applicable Age: {scheme['Applicable Age']}, Gender: {scheme['Gender']}, Income Range: {scheme['Income Range']}"
    }

# Function to check eligibility based on criteria
def check_eligibility(age, gender, state, income):
    eligible_schemes = []
    for _, scheme in df[df['State'] == state].iterrows():
        age_range = scheme['Applicable Age'].split('-')
        
        if len(age_range) == 2:
            min_age, max_age = map(int, age_range)
            if not (min_age <= age <= max_age):
                continue
        elif scheme['Applicable Age'] != 'All Ages' and int(scheme['Applicable Age']) != age:
            continue

        if scheme['Gender'] != 'Both' and scheme['Gender'] != gender:
            continue

        if scheme['State'] != state:
            continue

        if scheme['Income Range'] != 'No Income Limit':
            max_income = int(scheme['Income Range'].split()[2].replace(',', '')) * 100000
            if income > max_income:
                continue

        eligible_schemes.append(scheme['Scheme Name'])

    return eligible_schemes

# Function to get machine learning recommendations
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

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/check-schemes', methods=['GET', 'POST'])
def check_schemes():
    if request.method == 'POST':
        try:
            # Get form data
            categories = request.form.getlist('categories')
            # Remove any duplicates while preserving order
            categories = list(dict.fromkeys(categories))
            
            user_profile = {
                'age': int(request.form.get('age', 0)),
                'occupation': request.form.get('occupation', ''),
                'income': float(request.form.get('income', 0)),
                'state': request.form.get('state', ''),
                'location': request.form.get('location', ''),
                'category_preferences': categories
            }
            
            # Validate input
            if not user_profile['occupation'] or not user_profile['location'] or not user_profile['state']:
                return render_template('check_schemes.html', 
                                    error="Please fill in all required fields",
                                    recommended_schemes=[])
            
            # Get recommended schemes
            recommended_schemes = recommend_schemes(user_profile)
            
            if not recommended_schemes:
                default_images = [
                    "https://via.placeholder.com/350x200?text=Default+1",
                    "https://via.placeholder.com/350x200?text=Default+2",
                    "https://via.placeholder.com/350x200?text=Default+3",
                    "https://via.placeholder.com/350x200?text=Default+4",
                    "https://via.placeholder.com/350x200?text=Default+5",
                    "https://via.placeholder.com/350x200?text=Default+6"
                ]
                return render_template('recommendations.html', 
                                    error="No schemes found matching your criteria",
                                    recommended_schemes=[],
                                    default_images=default_images)
            
            # Sort schemes by score in descending order
            recommended_schemes.sort(key=lambda x: x.score, reverse=True)
            
            default_images = [
                "https://img.freepik.com/premium-vector/hand-drawn-india-map-illustration_23-2151716454.jpg?semt=ais_hybrid&w=740",
                "https://img.freepik.com/premium-vector/hand-drawn-india-map-illustration_23-2151716454.jpg?semt=ais_hybrid&w=740",
                "https://img.freepik.com/premium-vector/hand-drawn-india-map-illustration_23-2151716454.jpg?semt=ais_hybrid&w=740",
                "https://img.freepik.com/premium-vector/hand-drawn-india-map-illustration_23-2151716454.jpg?semt=ais_hybrid&w=740",
                "https://img.freepik.com/premium-vector/hand-drawn-india-map-illustration_23-2151716454.jpg?semt=ais_hybrid&w=740",
                "https://img.freepik.com/premium-vector/hand-drawn-india-map-illustration_23-2151716454.jpg?semt=ais_hybrid&w=740"
            ]
            return render_template('recommendations.html', 
                                recommended_schemes=recommended_schemes,
                                error=None,
                                default_images=default_images)
            
        except Exception as e:
            logger.error(f"Error in check_schemes: {str(e)}")
            default_images = [
                "https://via.placeholder.com/350x200?text=Default+1",
                "https://via.placeholder.com/350x200?text=Default+2",
                "https://via.placeholder.com/350x200?text=Default+3",
                "https://via.placeholder.com/350x200?text=Default+4",
                "https://via.placeholder.com/350x200?text=Default+5",
                "https://via.placeholder.com/350x200?text=Default+6"
            ]
            return render_template('recommendations.html', 
                                error="An error occurred while processing your request",
                                recommended_schemes=[],
                                default_images=default_images)
    
    return render_template('check_schemes.html', error=None, recommended_schemes=[])

@app.route('/scheme/<int:scheme_id>')
def scheme_details(scheme_id):
    try:
        scheme = get_scheme_by_id(scheme_id)
        if scheme:
            return render_template('scheme_details.html', scheme=scheme)
        return redirect(url_for('check_schemes'))
    except Exception as e:
        logger.error(f"Error in scheme_details: {str(e)}")
        return redirect(url_for('check_schemes'))

# Route for serving the chatbot page
@app.route("/chatbot")
def chatbot():
    logger.info("Accessing chatbot page")
    return render_template("chatindex.html")

# Route for handling chatbot messages
@app.route("/chatbot/send_message", methods=["POST"])
def send_message():
    try:
        user_input = request.json.get("message")
        if not user_input:
            logger.warning("No message provided in request")
            return jsonify({"response": "No input provided."})

        logger.info(f"Received message: {user_input}")

        # Send user input to the AI model
        response = chat_session.send_message(user_input)
        model_response = response.text

        logger.info(f"Model response: {model_response}")

        # Append conversation to chat history
        chat_session.history.append({"role": "user", "parts": [user_input]})
        chat_session.history.append({"role": "model", "parts": [model_response]})

        return jsonify({"response": model_response})
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

@app.route('/govt-schemes', methods=['GET', 'POST'])
def govt_schemes():
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
    ]
    scheme_details = None
    error = None
    selected_type = None
    selected_state = None
    scheme_name = None
    if request.method == 'POST':
        selected_type = request.form.get('scheme_type')
        selected_state = request.form.get('state')
        scheme_name = request.form.get('scheme_name')
        if not scheme_name:
            error = "Please enter a scheme name."
        else:
            # Use get_scheme_details to fetch details
            scheme_details = get_scheme_details(scheme_name)
            if not scheme_details:
                error = f"No details found for scheme: {scheme_name}"
    return render_template(
        'govt_schemes.html',
        states=states,
        scheme_details=scheme_details,
        error=error,
        selected_type=selected_type,
        selected_state=selected_state,
        scheme_name=scheme_name
    )

@app.route('/index2', methods=['GET', 'POST'])
def index2():
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
    ]
    scheme_details = None
    error = None
    selected_type = None
    selected_state = None
    scheme_name = None
    if request.method == 'POST':
        selected_type = request.form.get('scheme_type')
        selected_state = request.form.get('state')
        scheme_name = request.form.get('scheme_name')
        if not scheme_name:
            error = "Please enter a scheme name."
        else:
            scheme_details = get_scheme_details(scheme_name)
            if not scheme_details:
                error = f"No details found for scheme: {scheme_name}"
    return render_template(
        'index2.html',
        states=states,
        scheme_details=scheme_details,
        error=error,
        selected_type=selected_type,
        selected_state=selected_state,
        scheme_name=scheme_name
    )

if __name__ == '__main__':
    logger.info("Starting Flask application")
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)
