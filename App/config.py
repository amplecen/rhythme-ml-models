from pathlib import Path
import os
from dotenv import load_dotenv
from App.__init__ import __version__

load_dotenv()

#App Secret Key 
API_SECRET = os.environ.get("API_SECRET")

APP_TITLE = "Habit Completion Prediction API"
APP_VERSION = __version__
APP_DESCRIPTION = """ Predict whether a user will complete their habit based on historical data.

Features:
- 7 input features
- Returns probability and prediction
- Model accuracy: 60.0%"""

FEATURE_NAMES =[
    'day_of_week',
    'is_weekend',
    'current_streak',
    'completion_rate_7d',
    'completion_rate_30d',
    'days_since_start',
    'frequency_encoded'
]


#Model Loading 
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Models" / "Model_1" / "habit_model.pkl"

#Hugging Face Token and Model Used
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN nor fount in .env file")

#Sentiment Setting 
CONFIDENCE_THRESHOLS = 0.85
ROBERTA_RETRIES = 3
RETRY_DELAY = 2

#Behavioral Pattern Minimum Cnfiguration 
MIN_DAYS = 14
THRESHOLD = 0.35
