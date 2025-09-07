"""
Configuration settings for Crop Advisor ChatBot.
Centralizes all configurable parameters for easy maintenance.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data paths
DATA_DIR = BASE_DIR / "data"
CROPS_DB_PATH = DATA_DIR / "crops_database.csv"
REGIONAL_DATA_PATH = DATA_DIR / "regional_data.csv"
SOIL_DATA_PATH = DATA_DIR / "soil_data.csv"

# Models paths
MODELS_DIR = BASE_DIR / "models"
DISEASE_CLASSIFIER_PATH = MODELS_DIR / "disease_classifier.pkl"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.faiss"

# NLP model settings
HUGGINGFACE_MODEL = "distilbert-base-uncased"  # Lightweight model for free tier
MAX_INPUT_LENGTH = 128  # Maximum input length for the model

# FAISS settings
EMBEDDING_DIMENSION = 768  # Dimension of embeddings
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score for matches

# Application settings
DISCLAIMER = "This is AI generated solution. Please clarify with Agricultural Officers or Krishi Vigyan Kendra for professional advice."
DEFAULT_RESPONSE = "I'm sorry, I couldn't understand your query. Please provide both crop name and symptoms for accurate diagnosis."

# Telegram bot settings - ADD YOUR TOKEN HERE
TELEGRAM_BOT_TOKEN = "8188279363:AAFRMuDoh8f1mOudOUK3c3b7ECBx9tni0t4"

# Default region setting for Indian context
DEFAULT_REGION = "Karnataka"
DEFAULT_STATE = "Karnataka"

# Admin settings (for history reset functionality)
ADMIN_USER_IDS = ["admin_user_id_1", "admin_user_id_2"]  # Replace with actual admin user IDs