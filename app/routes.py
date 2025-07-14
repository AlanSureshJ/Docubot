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
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from fuzzywuzzy import process
from PIL import Image
import queue
import threading
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')

bp = Blueprint("routes", __name__)

# HuggingFace Pipeline Setup
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    framework="pt"
)

spacy_nlp = spacy.load("en_core_web_sm")

with open("intents.json", "r") as file:
    intents = json.load(file)

pdf_text_cache = ""
pdf_images_cache = []
request_queue = queue.Queue()

@bp.route("/")
def index():
    return render_template("index.html")

def process_pdf(pdf_file):
    global pdf_images_cache
    pdf_images_cache = []
    reader = PdfReader(pdf_file)
    extracted_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    for page in reader.pages:
        for img_file in page.images:
            image_data = img_file.data
            img = Image.open(io.BytesIO(image_data))
            pdf_images_cache.append(img)
            extracted_text += "\n[Image Analysis]: [Image detected - analysis will be performed on request]"

    return extracted_text if extracted_text else "Unable to extract text from this PDF."

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def analyze_image(image):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ocr_result = pytesseract.image_to_string(thresh, config="--psm 6")
    if not ocr_result.strip():
        ocr_result = "No handwritten text detected."
    return f"OCR Result: {ocr_result}"

def get_relevant_text(query, pdf_text):
    parser = PlaintextParser.from_string(pdf_text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 5)
    return " ".join(str(sentence) for sentence in summary)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

def summarize_text(text, num_sentences=5):
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

def generate_quiz(text, num_questions=3):
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
        choices = list(set([missing_word] + random.sample(words, min(3, len(words) - 1))))
        while len(choices) < 4:
            choices.append(random.choice(words))
        random.shuffle(choices)
        questions.append({"question": question, "choices": choices, "answer": missing_word})
    return questions

def get_intent(text):
    text = text.lower()
    greetings = ["hello", "hi", "hey", "greetings"]
    if text in greetings:
        return "greeting", "Hello! How can I assist you today?"
    for intent in intents["intents"]:
        if text in [pattern.lower() for pattern in intent["patterns"]]:
            return intent["tag"], random.choice(intent["responses"])
    all_patterns = {pattern: intent["tag"] for intent in intents["intents"] for pattern in intent["patterns"]}
    best_match, score = process.extractOne(text, list(all_patterns.keys()))
    if score > 70:
        matched_tag = all_patterns[best_match]
        for intent in intents["intents"]:
            if intent["tag"] == matched_tag:
                return matched_tag, random.choice(intent["responses"])
    return None, "I'm not sure how to respond to that. Can you rephrase?"

def is_greeting(text):
    base_greetings = ["hello", "hi", "hey", "greetings"]
    best_match, score = process.extractOne(text, base_greetings)
    return score > 70, best_match if score > 70 else None

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def is_question(text):
    question_words = ["what", "why", "how", "when", "where", "who", "is", "are", "do", "does", "can", "should"]
    words = nltk.word_tokenize(text.lower())
    return words[0] in question_words if words else False

def is_pdf_query(text):
    pdf_keywords = ["pdf", "document", "upload", "file", "analyze", "extract", "summarize", "quiz"]
    words = nltk.word_tokenize(text.lower())
    return any(word in words for word in pdf_keywords)

@bp.route("/chat", methods=["POST"])
def chat():
    global pdf_text_cache
    user_input = request.json.get("message", "").strip().lower()
    is_greet, matched_greeting = is_greeting(user_input)
    if is_greet:
        return jsonify({"response": f"{matched_greeting.capitalize()}! How can I assist you today?"})
    if is_pdf_query(user_input):
        if "summarize" in user_input and pdf_text_cache:
            return jsonify({"response": summarize_text(pdf_text_cache)})
        elif "quiz" in user_input and pdf_text_cache:
            return jsonify({"response": generate_quiz(pdf_text_cache)})
        else:
            return jsonify({"response": "You can upload a PDF, and I'll analyze it for you!"})
    intent, response = get_intent(user_input)
    if response:
        return jsonify({"response": response})
    return jsonify({"response": "I'm not sure how to respond to that. Can you rephrase?"})

@bp.route("/upload", methods=["POST"])
def upload_pdf():
    global pdf_text_cache, pdf_images_cache
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    pdf_file = request.files["pdf"]
    pdf_text_cache = process_pdf(pdf_file)
    return jsonify({"message": "PDF uploaded successfully", "redirect": True})

@bp.route("/ask", methods=["POST"])
def ask():
    global pdf_text_cache, pdf_images_cache
    data = request.get_json()
    question = data.get("question", "").strip()
    if not pdf_text_cache:
        return jsonify({"answer": "Please upload a PDF first."})
    lower_question = question.lower()
    if "summary" in lower_question or "summarize" in lower_question:
        return jsonify({"answer": summarize_text(pdf_text_cache)})
    if "quiz" in lower_question or "question" in lower_question:
        return jsonify({"answer": generate_quiz(pdf_text_cache)})
    relevant_text = pdf_text_cache[:3000]
    try:
        response = qa_pipeline({"question": question, "context": relevant_text})
        text_response = response["answer"]
        image_response = ""
        if any(word in lower_question for word in ["image", "picture", "diagram", "handwriting", "graph"]):
            if pdf_images_cache:
                image_response = analyze_image(pdf_images_cache[0])
        combined_response = text_response
        if image_response:
            combined_response += f"\n\n[Image Insight]: {image_response}"
        return jsonify({"answer": combined_response})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})
