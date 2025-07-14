# Docubot

Docubot is an intelligent PDF assistant that allows you to upload documents and interact with them using natural language. It supports summarization, question answering, quiz generation and image analysis.

## Features

- Upload and analyze PDF documents
- Chat-based interaction with the uploaded document
- Summarize document content using a transformer model
- Generate quiz questions based on document content
- Extract text from embedded images using OCR
- Handle both text and image-based PDF content

## Technologies Used

- Python
- Flask
- Hugging Face Transformers
- PyTorch
- spaCy
- NLTK
- OpenCV and Tesseract OCR
- PDF parsing with PyPDF
- Text summarization using BART
- Question answering using RoBERTa

## Installation

1. Clone the repository


git clone https://github.com/your-username/docubot.git
cd docubot

Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate 
Install dependencies


pip install -r requirements.txt
Download required models and NLTK data

python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
Usage
Start the Flask server:


python app.py
Open your browser and visit http://localhost:5000

Project Structure

Copy
Edit
docubot/
│
├── app.py               # Flask app entry point
├── app/
│   ├── routes.py        # Main backend logic
│   └── templates/
│       └── index.html   # Frontend page
├── intents.json         # Chatbot intent logic
├── requirements.txt     # Python dependencies
