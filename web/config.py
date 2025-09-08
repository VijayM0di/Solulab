
# AI Safety Models Configuration
import os
from datetime import datetime

class Config:
    # Model Configuration
    MODEL_CONFIGS = {
        'abuse_detection': {
            'model_name': 'bert-base-uncased',
            'max_length': 512,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'epochs': 3,
            'labels': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        },
        'escalation_detection': {
            'model_name': 'distilbert-base-uncased',
            'sequence_length': 256,
            'threshold_aggressive': 0.7,
            'threshold_escalating': 0.8,
            'window_size': 5  # Number of messages to analyze for patterns
        },
        'crisis_intervention': {
            'model_name': 'bert-base-uncased',
            'max_length': 512,
            'crisis_keywords': [
                'suicide', 'kill myself', 'end my life', 'self-harm', 'hurt myself',
                'not worth living', 'better off dead', 'want to die'
            ],
            'severity_levels': ['low', 'medium', 'high', 'critical']
        },
        'content_filtering': {
            'age_categories': {
                'child': {'min_age': 0, 'max_age': 12},
                'teen': {'min_age': 13, 'max_age': 17},
                'adult': {'min_age': 18, 'max_age': 100}
            },
            'content_categories': [
                'violence', 'sexual', 'drugs', 'profanity', 'mature_themes'
            ]
        }
    }

    # Data Paths
    DATA_DIR = 'data'
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = 'models/saved'

    # API Configuration
    API_HOST = '0.0.0.0'
    API_PORT = 8000
    API_VERSION = 'v1'

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Safety Thresholds
    SAFETY_THRESHOLDS = {
        'abuse_threshold': 0.7,
        'escalation_threshold': 0.8,
        'crisis_threshold': 0.9,
        'content_filter_threshold': 0.6
    }

    # Real-time Processing
    BATCH_PROCESSING = True
    REAL_TIME_BUFFER_SIZE = 100

    # Model Performance Metrics
    EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    # Database (if needed for production)
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///ai_safety.db')

    # External Services
    CRISIS_HOTLINE_API = os.getenv('CRISIS_HOTLINE_API')
    MODERATION_WEBHOOK = os.getenv('MODERATION_WEBHOOK')

    @staticmethod
    def get_model_path(model_name):
        return os.path.join(Config.MODELS_DIR, f"{model_name}_model.pkl")

    @staticmethod
    def get_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
