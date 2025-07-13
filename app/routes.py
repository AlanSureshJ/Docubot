import io
import base64
from io import BytesIO
import random
import json
import time
import nltk
import numpy as np
import spacy
import pytesseract
import cv2
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from flask import Blueprint, request, jsonify, render_template
from pypdf import PdfReader
import google.generativeai as genai
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from fuzzywuzzy import process
from PIL import Image
import queue
import threading

nltk.download('punkt_tab')

# Initialize Flask Blueprint
bp = Blueprint("routes", __name__)

# Gemini API Setup
genai.configure(api_key="AIzaSyA3YVkdBL0R1RcBOJV9Y1dmE4FxCg_8_2I")

# Load NLP Resources

spacy_nlp = spacy.load("en_core_web_sm")

# Load Intents JSON
with open("intents.json", "r") as file:
    intents = json.load(file)

# Global Variable to Store PDF Text
pdf_text_cache = ""
pdf_images_cache = []

request_queue = queue.Queue()


@bp.route("/")
def index():
    return render_template("index.html")

def process_pdf(pdf_file):
    """Extracts text and images from a PDF file, including OCR for handwritten text."""
    global pdf_images_cache
    pdf_images_cache = []  # Reset image cache
    reader = PdfReader(pdf_file)

    # Extract text
    extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Extract images and run OCR
    for page in reader.pages:
        for img_file in page.images:
            image_data = img_file.data  # Extract raw image data
            img = Image.open(io.BytesIO(image_data))  # Convert to PIL image
            pdf_images_cache.append(img)  # Store extracted images

            # Analyze image using Gemini and OCR
            image_analysis = analyze_image(img)
            extracted_text += f"\n[Image Analysis]: {image_analysis}"

    return extracted_text if extracted_text else "Unable to extract text from this PDF."

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()



def analyze_image(image):
    """Analyzes an image using Gemini Pro and extracts handwritten text using OCR."""
    
    # Gemini API Analysis with rate limit handling
    model = genai.GenerativeModel("gemini-1.5-pro")
    encoded_image = encode_image(image)  # Convert image to Base64

    try:
        response = model.generate_content([
            {"inline_data": {"mime_type": "image/png", "data": encoded_image}}
        ])
        gemini_result = response.text if response and hasattr(response, "text") else "Couldn't analyze this image."
    except Exception as e:
        gemini_result = f"Error in Gemini analysis: {e}"
    
    time.sleep(10)  # Pause to avoid exceeding the rate limit

    # OCR for Handwritten Text
    image_np = np.array(image)  # Convert PIL to NumPy
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Improve contrast

    ocr_result = pytesseract.image_to_string(thresh, config="--psm 6")  # Extract text

    if not ocr_result.strip():
        ocr_result = "No handwritten text detected."

    # Combine results
    return f"Gemini Analysis: {gemini_result}\nHandwritten OCR: {ocr_result}"

    
def get_relevant_text(query, pdf_text):
    parser = PlaintextParser.from_string(pdf_text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 5)  # 5 key sentences
    return " ".join(str(sentence) for sentence in summary)


def summarize_text(text, num_sentences=5):
    """Summarizes text using the TextRank algorithm."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)



def generate_quiz(text, num_questions=3):
    """Generates multiple-choice questions from text."""
    sentences = nltk.sent_tokenize(text)
    questions = []

    for _ in range(min(num_questions, len(sentences))):
        sentence = random.choice(sentences)
        sentences.remove(sentence)

        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word.isalpha()]
        if len(words) < 4:
            continue

        missing_word = random.choice(words)
        question = sentence.replace(missing_word, "____")

        # Ensure choices are unique and contain at least 4 options
        choices = list(set([missing_word] + random.sample(words, min(3, len(words)-1))))
        while len(choices) < 4:
            choices.append(random.choice(words))  # Fill up to 4 choices if needed
        random.shuffle(choices)

        questions.append({
            "question": question,
            "choices": choices,
            "answer": missing_word
        })

    return questions

def get_intent(text):
    """Detects intent using exact and fuzzy matching on patterns."""
    text = text.lower()

    greetings = ["hello", "hi", "hey", "greetings"]
    if text in greetings:
        return "greeting", "Hello! How can I assist you today?"
    
    # Check for exact match in patterns
    for intent in intents["intents"]:
        if text in [pattern.lower() for pattern in intent["patterns"]]:
            return intent["tag"], random.choice(intent["responses"])
    
    # Flatten patterns for fuzzy matching
    all_patterns = {pattern: intent["tag"] for intent in intents["intents"] for pattern in intent["patterns"]}
    
    # Find best match among all patterns
    best_match, score = process.extractOne(text, list(all_patterns.keys()))

    if score > 70:  # Adjust threshold as needed
        matched_tag = all_patterns[best_match]
        for intent in intents["intents"]:
            if intent["tag"] == matched_tag:
                return matched_tag, random.choice(intent["responses"])

    return None, "I'm not sure how to respond to that. Can you rephrase?"

def is_greeting(text):
    """Detects if input is a greeting using fuzzy matching."""
    base_greetings = ["hello", "hi", "hey", "greetings"]

    best_match, score = process.extractOne(text, base_greetings)
    return score > 70, best_match if score > 70 else None

def get_synonyms(word):
    """Fetches synonyms using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def is_question(text):
    """Detects if input is a question."""
    question_words = ["what", "why", "how", "when", "where", "who", "is", "are", "do", "does", "can", "should"]
    words = nltk.word_tokenize(text.lower())
    return words[0] in question_words if words else False

def is_pdf_query(text):
    """Detects if input is related to PDFs."""
    pdf_keywords = ["pdf", "document", "upload", "file", "analyze", "extract", "summarize", "quiz"]
    words = nltk.word_tokenize(text.lower())
    return any(word in words for word in pdf_keywords)

@bp.route("/chat", methods=["POST"])
def chat():
    """Handles chatbot interactions."""
    global pdf_text_cache
    user_input = request.json.get("message", "").strip().lower()

    # Check for greetings first
    is_greet, matched_greeting = is_greeting(user_input)
    if is_greet:
        return jsonify({"response": f"{matched_greeting.capitalize()}! How can I assist you today?"})

    # Check for PDF-related queries
    if is_pdf_query(user_input):
        if "summarize" in user_input and pdf_text_cache:
            return jsonify({"response": summarize_text(pdf_text_cache)})
        elif "quiz" in user_input and pdf_text_cache:
            return jsonify({"response": generate_quiz(pdf_text_cache)})
        else:
            return jsonify({"response": "You can upload a PDF, and I'll analyze it for you!"})

    # Check for intents
    intent, response = get_intent(user_input)
    if response:
        return jsonify({"response": response})

    # Default fallback response
    return jsonify({"response": "I'm not sure how to respond to that. Can you rephrase?"})

@bp.route("/upload", methods=["POST"])
def upload_pdf():
    """Handles PDF uploads and extracts text & images."""
    global pdf_text_cache, pdf_images_cache

    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["pdf"]
    pdf_text_cache = process_pdf(pdf_file)

    return jsonify({"message": "PDF uploaded successfully", "redirect": True})

@bp.route("/ask", methods=["POST"])
def ask():
    """Handles user queries related to the uploaded PDF."""
    global pdf_text_cache, pdf_images_cache
    data = request.get_json()
    question = data.get("question")

    if not pdf_text_cache and not pdf_images_cache:
        return jsonify({"answer": "Please upload a PDF first."})

    relevant_text = get_relevant_text(question, pdf_text_cache)
    prompt = f"Based on this summary:\n{relevant_text}\nUser: {question}\nAssistant:"

    model = genai.GenerativeModel("gemini-1.5-pro")  # For image analysis


    try:
        # Get text-based answer
        try:
            response = model.generate_content(prompt)
            text_response = response.text if response and hasattr(response, "text") else "I couldn't find an answer."
        except Exception as e:
            text_response = f"Error: {str(e)}"

        # Analyze first image if available
        image_response = ""
        if pdf_images_cache:
            image_response = analyze_image(pdf_images_cache[0])  # Analyze first image

        combined_response = text_response
        if image_response:
            combined_response += f"\nAdditionally, based on the image in the document:\n{image_response}"

        return jsonify({"answer": combined_response})

    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})

