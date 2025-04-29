import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Configure the model
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

class Scheme:
    def __init__(self, id, name, short_description, description, image_url, eligibility_criteria, benefits, required_documents, apply_url, categories):
        self.id = id
        self.name = name
        self.short_description = short_description
        self.description = description
        self.image_url = image_url
        self.eligibility_criteria = eligibility_criteria
        self.benefits = benefits
        self.required_documents = required_documents
        self.apply_url = apply_url
        self.categories = categories
        self.score = None
        self.reason = None

def get_gemini_model():
    """Get configured Gemini model with proper settings"""
    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        return model
    except Exception as e:
        logger.error(f"Error initializing Gemini model: {str(e)}")
        return None

def parse_json_response(response_text: str) -> Optional[Dict]:
    """Safely parse JSON response from the API"""
    try:
        # Clean the response text
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # Parse JSON
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Response text: {response_text}")
        return None
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        return None

def get_scheme_details(scheme_name: str) -> Optional[Scheme]:
    """Get detailed information about a scheme using Gemini API"""
    model = get_gemini_model()
    if not model:
        return None
    prompt = f"""
    Provide detailed information about the government scheme: {scheme_name}
    Format the response as a JSON object with the following structure:
    {{
        "name": "Scheme Name",
        "short_description": "Brief description in one line",
        "description": "Detailed description",
        "image_url": "URL to scheme logo/image",
        "eligibility_criteria": ["List of eligibility criteria"],
        "benefits": ["List of benefits"],
        "required_documents": ["List of required documents"],
        "apply_url": "Official application URL",
        "categories": ["List of relevant categories"]
    }}
    
    Make sure to:
    1. Include accurate and up-to-date information
    2. List specific eligibility criteria
    3. Provide comprehensive benefits
    4. List all required documents
    5. Include the official application URL
    6. Add relevant categories (e.g., healthcare, education, housing, etc.)
    
    Return ONLY the JSON object, no additional text.
    """
    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            logger.error("Empty response from Gemini API")
            return None
        scheme_data = parse_json_response(response.text)
        if not scheme_data:
            return None
        expected_fields = [
            "name", "short_description", "description", "image_url",
            "eligibility_criteria", "benefits", "required_documents", "apply_url", "categories"
        ]
        for field in expected_fields:
            if field not in scheme_data:
                logger.error(f"Missing expected field '{field}' in scheme data: {scheme_data}")
                return None
        scheme = Scheme(
            id=hash(scheme_name) % 10000,
            name=scheme_data["name"],
            short_description=scheme_data["short_description"],
            description=scheme_data["description"],
            image_url=scheme_data["image_url"],
            eligibility_criteria=scheme_data["eligibility_criteria"],
            benefits=scheme_data["benefits"],
            required_documents=scheme_data["required_documents"],
            apply_url=scheme_data["apply_url"],
            categories=scheme_data["categories"]
        )
        return scheme
    except Exception as e:
        logger.error(f"Error getting scheme details: {str(e)}")
        return None

def recommend_schemes(user_profile: Dict) -> List[Scheme]:
    """
    Recommend schemes based on user profile using Gemini API
    user_profile should contain:
    - age
    - occupation
    - income
    - state
    - location (urban/rural)
    - category_preferences
    """
    model = get_gemini_model()
    if not model:
        return []
    prompt = f"""
    Based on the following user profile, recommend suitable government schemes:
    Age: {user_profile.get('age')}
    Occupation: {user_profile.get('occupation')}
    Annual Income: ₹{user_profile.get('income')}
    State: {user_profile.get('state')}
    Location: {user_profile.get('location')}
    Interested Categories: {', '.join(user_profile.get('category_preferences', []))}
    
    For each recommended scheme, provide:
    1. Scheme name
    2. Brief description
    3. Relevance score (0-10)
    4. Why it's recommended
    
    Format the response as a JSON array of objects:
    [
        {{
            "name": "Scheme Name",
            "description": "Brief description",
            "score": relevance_score,
            "reason": "Why it's recommended"
        }},
        ...
    ]
    
    Consider:
    - Income eligibility
    - Age-appropriate schemes
    - Location-specific schemes (both state and urban/rural)
    - Occupation-related schemes
    - Category preferences
    - State-specific schemes and benefits
    
    Important:
    1. Return at least 6 different schemes if possible
    2. Include both central and state government schemes
    3. Ensure schemes are relevant to the user's profile
    4. Provide accurate and up-to-date information
    5. Include schemes from all selected categories
    
    Return ONLY the JSON array, no additional text.
    """
    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            logger.error("Empty response from Gemini API")
            return []
        recommendations = parse_json_response(response.text)
        if not recommendations:
            return []
        recommended_schemes = []
        for idx, rec in enumerate(recommendations):
            if len(recommended_schemes) >= 6:
                break
            if idx < 3:
                # Fetch full details for the first 3
                scheme = get_scheme_details(rec["name"])
                if scheme:
                    scheme.score = rec.get("score", 0)
                    scheme.reason = rec.get("reason", "")
                    recommended_schemes.append(scheme)
            else:
                # Only show name, score, and reason for the next 3
                scheme = Scheme(
                    id=hash(rec["name"]) % 10000,
                    name=rec["name"],
                    short_description=rec.get("description", ""),
                    description="",
                    image_url="",
                    eligibility_criteria=[],
                    benefits=[],
                    required_documents=[],
                    apply_url="",
                    categories=[]
                )
                scheme.score = rec.get("score", 0)
                scheme.reason = rec.get("reason", "")
                recommended_schemes.append(scheme)
        return recommended_schemes
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return []

def get_scheme_by_id(scheme_id: int) -> Optional[Scheme]:
    """Get scheme details by ID using Gemini API"""
    model = get_gemini_model()
    if not model:
        return None
        
    prompt = f"""
    Find a government scheme with ID: {scheme_id}
    Provide detailed information about this scheme.
    Format the response as a JSON object with the following structure:
    {{
        "name": "Scheme Name",
        "short_description": "Brief description in one line",
        "description": "Detailed description",
        "image_url": "URL to scheme logo/image",
        "eligibility_criteria": ["List of eligibility criteria"],
        "benefits": ["List of benefits"],
        "required_documents": ["List of required documents"],
        "apply_url": "Official application URL",
        "categories": ["List of relevant categories"]
    }}
    
    Return ONLY the JSON object, no additional text.
    """
    
    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            logger.error("Empty response from Gemini API")
            return None
            
        scheme_data = parse_json_response(response.text)
        if not scheme_data:
            return None
            
        return Scheme(
            id=scheme_id,
            name=scheme_data["name"],
            short_description=scheme_data["short_description"],
            description=scheme_data["description"],
            image_url=scheme_data["image_url"],
            eligibility_criteria=scheme_data["eligibility_criteria"],
            benefits=scheme_data["benefits"],
            required_documents=scheme_data["required_documents"],
            apply_url=scheme_data["apply_url"],
            categories=scheme_data["categories"]
        )
    except Exception as e:
        logger.error(f"Error getting scheme by ID: {str(e)}")
        return None

# Scheme database
schemes = [
    Scheme(
        id=1,
        name="Pradhan Mantri Jan Arogya Yojana",
        short_description="Healthcare coverage for underprivileged families",
        description="PM-JAY is the world's largest health insurance scheme providing coverage up to ₹5 lakhs per family per year for secondary and tertiary care hospitalization.",
        image_url="https://presentations.gov.in/wp-content/uploads/2020/06/PMJAY-Preview.png",
        eligibility_criteria=[
            "Families included in SECC database",
            "No pucca house",
            "No adult member between 16-59 years",
            "No adult member with a job",
            "No motorized vehicle",
            "No landline phone",
            "No refrigerator",
            "No agricultural land"
        ],
        benefits=[
            "Coverage up to ₹5 lakhs per family per year",
            "No restriction on family size",
            "All pre-existing conditions covered from day one",
            "Cashless treatment at empaneled hospitals"
        ],
        required_documents=[
            "Aadhaar Card",
            "Ration Card",
            "BPL Certificate",
            "Income Certificate",
            "Family Photograph"
        ],
        apply_url="https://www.myscheme.gov.in/schemes/ab-pmjay",
        categories=["healthcare", "insurance", "medical"]
    ),
    Scheme(
        id=2,
        name="Pradhan Mantri Awas Yojana",
        short_description="Affordable housing for urban and rural poor",
        description="PMAY aims to provide housing for all by 2022, focusing on affordable housing for urban and rural poor through various components.",
        image_url="https://upload.wikimedia.org/wikipedia/en/6/6c/Pradhan_Mantri_Awas_Yojana-Urban_%28PMAY-U%29_logo.png",
        eligibility_criteria=[
            "Family income below ₹3 lakhs per annum",
            "No pucca house",
            "Not availed any housing scheme from Government",
            "First-time home buyer"
        ],
        benefits=[
            "Interest subsidy on home loans",
            "Affordable housing units",
            "Credit-linked subsidy",
            "Affordable housing projects"
        ],
        required_documents=[
            "Aadhaar Card",
            "Income Certificate",
            "Property Documents",
            "Bank Statement",
            "Employment Certificate"
        ],
        apply_url="https://www.myscheme.gov.in/schemes/pmay-g",
        categories=["housing", "infrastructure", "urban"]
    ),
    # Add more schemes here...
] 