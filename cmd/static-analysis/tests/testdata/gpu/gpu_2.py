from transformers import pipeline
import json

# Load model on cold start
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
