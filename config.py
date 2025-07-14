import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or 'your_gemini_api_key'
