from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask import make_response
from flask import g
import pandas as pd
import pickle
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
import logging
import uuid
import json
from datetime import datetime
from scheme_data import recommend_schemes, get_scheme_by_id, get_scheme_details

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key and log if it's available
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY/GOOGLE_API_KEY not found in environment variables")
else:
    logger.info("Gemini API key found in environment variables")

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
    "response_mime_type": "text/html",
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the model
try:
    selected_model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    model = genai.GenerativeModel(
        model_name=selected_model_name,
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction="You are trained to assist users in identifying the most suitable government scheme based on their requirements. Format the response in valid HTML with appropriate bold sections and even make the list headings in bold.",
    )
    logger.info(f"Successfully initialized Gemini model: {selected_model_name}")
except Exception as e:
    logger.error(f"Error initializing Gemini model: {str(e)}")

app = Flask(__name__)

# Language helper
@app.before_request
def set_language_context():
    lang = request.cookies.get('lang') or request.args.get('lang') or 'en'
    g.lang = 'hi' if str(lang).lower().startswith('hi') else 'en'

@app.route('/set-language', methods=['GET', 'POST'])
def set_language():
    lang = request.values.get('lang', 'en')
    lang = 'hi' if str(lang).lower().startswith('hi') else 'en'
    resp = make_response(redirect(request.referrer or url_for('index')))
    # 180 days
    resp.set_cookie('lang', lang, max_age=180*24*3600, samesite='Lax')
    return resp

# Tracking storage helpers
TRACKING_FILE = os.path.join(os.path.dirname(__file__), 'application_tracking.json')

def _load_tracking_store():
    if os.path.exists(TRACKING_FILE):
        try:
            with open(TRACKING_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_tracking_store(data):
    with open(TRACKING_FILE, 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/start-application')
def start_application():
    scheme_id = request.args.get('scheme_id')
    scheme_name = request.args.get('scheme_name') or ''
    if not scheme_id and not scheme_name:
        return redirect(url_for('index'))
    app_id = uuid.uuid4().hex[:8]
    entry = {
        'id': app_id,
        'scheme_id': scheme_id,
        'scheme_name': scheme_name,
        'status': 'Application Submitted',
        'history': [
            {'status': 'Application Submitted', 'timestamp': datetime.utcnow().isoformat() + 'Z'},
            {'status': 'Under Review', 'timestamp': None},
            {'status': 'Documents Verified', 'timestamp': None},
            {'status': 'Approval Pending', 'timestamp': None},
            {'status': 'Benefits Disbursed', 'timestamp': None},
        ],
        'created_at': datetime.utcnow().isoformat() + 'Z'
    }
    store = _load_tracking_store()
    store[app_id] = entry
    _save_tracking_store(store)
    return redirect(url_for('tracking_status', app_id=app_id))

@app.route('/tracking')
def tracking_lookup():
    app_id = request.args.get('id')
    if app_id:
        return redirect(url_for('tracking_status', app_id=app_id))
    return render_template('tracking_lookup.html')

@app.route('/tracking/<app_id>')
def tracking_status(app_id):
    store = _load_tracking_store()
    entry = store.get(app_id)
    if not entry:
        return render_template('tracking_status.html', not_found=True, app_id=app_id)
    return render_template('tracking_status.html', not_found=False, app=entry)

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
            
            # Get recommended schemes with language
            language = getattr(g, 'lang', 'en')
            recommended_schemes = recommend_schemes(user_profile, language=language)
            
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
            recommended_schemes.sort(key=lambda x: (x.score or 0), reverse=True)
            
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
    except google_exceptions.ResourceExhausted as e:
        message = str(e)
        match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", message)
        wait_seconds = int(match.group(1)) if match else 60
        logger.error(f"Rate limit exceeded: {message}")
        return jsonify({
            "response": f"Rate limit exceeded. Please wait about {wait_seconds} seconds and try again.",
            "retry_after_seconds": wait_seconds
        }), 429
    except google_exceptions.PermissionDenied as e:
        logger.error(f"Permission denied calling Gemini API: {str(e)}")
        return jsonify({"response": "Access to the requested model is denied. Please check your API key permissions and model availability."}), 403
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Gemini API error: {str(e)}")
        return jsonify({"response": "An error occurred calling the Gemini API. Please try again later."}), 502
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500

@app.route('/govt-schemes', methods=['GET', 'POST'])
def govt_schemes():
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal"
    ]
    scheme_details_data = None
    error = None
    selected_type = None
    selected_state = None
    scheme_name = None
    selected_language = getattr(g, 'lang', 'en')
    if request.method == 'POST':
        selected_type = request.form.get('scheme_type')
        selected_state = request.form.get('state')
        scheme_name = request.form.get('scheme_name')
        if not scheme_name:
            error = "Please enter a scheme name."
        else:
            # Fetch details in the selected language, considering scheme type/state
            scheme_details_data = get_scheme_details(
                scheme_name,
                language=selected_language,
                scheme_type=selected_type,
                state=selected_state
            )
            if not scheme_details_data:
                error = f"No details found for scheme: {scheme_name}"
    return render_template(
        'govt_schemes.html',
        states=states,
        scheme_details=scheme_details_data,
        error=error,
        selected_type=selected_type,
        selected_state=selected_state,
        scheme_name=scheme_name,
        selected_language=selected_language
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
