
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import json

class CrisisInterventionModel:
    """
    Advanced Crisis Intervention Detection System
    Detects severe emotional distress, self-harm indicators, and suicidal ideation
    Provides appropriate intervention recommendations and crisis resources
    """

    def __init__(self, crisis_threshold: float = 0.8):
        self.crisis_threshold = crisis_threshold
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.crisis_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
        self.is_trained = False

        # Crisis indicators and keywords
        self.crisis_keywords = {
            'suicide': [
                'kill myself', 'end my life', 'commit suicide', 'take my own life',
                'not want to live', 'better off dead', 'end it all', 'suicide',
                'suicidal thoughts', 'want to die', 'wish I was dead', 'kill me'
            ],
            'self_harm': [
                'cut myself', 'hurt myself', 'self harm', 'self-harm', 'cutting',
                'burn myself', 'harm myself', 'injure myself', 'self injury',
                'self mutilation', 'scratch myself', 'hit myself'
            ],
            'hopelessness': [
                'no hope', 'hopeless', 'pointless', 'meaningless', 'worthless',
                'no point', 'give up', 'cant go on', "can't take it", 'no future',
                'nothing matters', 'no way out', 'trapped', 'stuck'
            ],
            'despair': [
                'desperate', 'despair', 'overwhelmed', 'drowning', 'suffocating',
                'cant breathe', 'crushing', 'unbearable', 'too much', 'breaking down',
                'falling apart', 'cant cope', 'losing it'
            ],
            'isolation': [
                'alone', 'lonely', 'nobody cares', 'no one understands', 'isolated',
                'abandoned', 'rejected', 'unwanted', 'unloved', 'forgotten',
                'no friends', 'no family', 'nobody would miss me'
            ],
            'plans': [
                'have a plan', 'planning to', 'going to kill', 'tonight', 'today',
                'pills', 'rope', 'gun', 'bridge', 'method', 'way to die',
                'final decision', 'last time', 'goodbye'
            ]
        }

        # Severity levels
        self.severity_levels = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.95
        }

        # Crisis resources
        self.crisis_resources = {
            'suicide_hotlines': {
                'US': {
                    'name': '988 Suicide & Crisis Lifeline',
                    'phone': '988',
                    'text': 'Text HOME to 741741',
                    'web': 'https://suicidepreventionlifeline.org'
                },
                'UK': {
                    'name': 'Samaritans',
                    'phone': '116 123',
                    'email': 'jo@samaritans.org',
                    'web': 'https://www.samaritans.org'
                },
                'international': {
                    'name': 'International Association for Suicide Prevention',
                    'web': 'https://www.iasp.info/resources/Crisis_Centres/'
                }
            },
            'emergency_services': '911 (US), 999 (UK), 112 (EU)',
            'text_support': {
                'crisis_text_line': 'Text HOME to 741741',
                'description': '24/7 support via text message'
            }
        }

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_crisis_features(self, text: str) -> Dict:
        """Extract features relevant to crisis detection"""
        features = {}
        text_lower = text.lower()

        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))

        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        features['sentiment_compound'] = sentiment['compound']
        features['sentiment_negative'] = sentiment['neg']
        features['sentiment_positive'] = sentiment['pos']
        features['sentiment_neutral'] = sentiment['neu']

        # TextBlob analysis
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity

        # Crisis keyword analysis
        total_crisis_score = 0
        for category, keywords in self.crisis_keywords.items():
            count = 0
            for keyword in keywords:
                if keyword in text_lower:
                    count += text_lower.count(keyword)
            features[f'{category}_keywords'] = count
            total_crisis_score += count

        features['total_crisis_keywords'] = total_crisis_score

        # Linguistic patterns
        features['first_person_pronouns'] = len(re.findall(r'\b(i|me|my|myself|mine)\b', text_lower))
        features['negative_words'] = self.count_negative_words(text_lower)
        features['question_marks'] = text.count('?')
        features['exclamation_marks'] = text.count('!')
        features['ellipsis'] = text.count('...')

        # Temporal indicators
        features['future_tense'] = len(re.findall(r'\b(will|going to|gonna|planning)\b', text_lower))
        features['past_tense'] = len(re.findall(r'\b(was|were|had|did|used to)\b', text_lower))

        # Emotional intensity
        features['intensity_words'] = self.count_intensity_words(text_lower)
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        return features

    def count_negative_words(self, text: str) -> int:
        """Count negative emotion words"""
        negative_words = [
            'sad', 'depressed', 'anxious', 'scared', 'afraid', 'worried', 'stressed',
            'tired', 'exhausted', 'empty', 'numb', 'broken', 'hurt', 'pain',
            'suffering', 'agony', 'misery', 'torment', 'anguish', 'grief'
        ]
        return sum(1 for word in negative_words if word in text)

    def count_intensity_words(self, text: str) -> int:
        """Count words indicating emotional intensity"""
        intensity_words = [
            'extremely', 'very', 'really', 'so', 'incredibly', 'absolutely',
            'completely', 'totally', 'entirely', 'utterly', 'deeply', 'severely'
        ]
        return sum(1 for word in intensity_words if word in text)

    def assess_crisis_severity(self, features: Dict) -> Dict:
        """Assess the severity level of potential crisis"""

        # Rule-based severity assessment
        severity_score = 0.0

        # High-risk keyword categories
        high_risk_categories = ['suicide', 'self_harm', 'plans']
        medium_risk_categories = ['hopelessness', 'despair']
        low_risk_categories = ['isolation']

        # Calculate keyword-based score
        for category in high_risk_categories:
            if features[f'{category}_keywords'] > 0:
                severity_score += 0.4 * min(features[f'{category}_keywords'], 3)  # Cap at 3 mentions

        for category in medium_risk_categories:
            if features[f'{category}_keywords'] > 0:
                severity_score += 0.2 * min(features[f'{category}_keywords'], 2)

        for category in low_risk_categories:
            if features[f'{category}_keywords'] > 0:
                severity_score += 0.1 * min(features[f'{category}_keywords'], 2)

        # Sentiment-based adjustments
        if features['sentiment_compound'] < -0.7:
            severity_score += 0.3
        elif features['sentiment_compound'] < -0.5:
            severity_score += 0.1

        # Linguistic pattern adjustments
        if features['first_person_pronouns'] > 5:  # High self-reference
            severity_score += 0.1

        if features['future_tense'] > 0 and features['total_crisis_keywords'] > 0:  # Planning indicators
            severity_score += 0.2

        # Normalize severity score
        severity_score = min(1.0, severity_score)

        # Determine severity level
        if severity_score >= self.severity_levels['critical']:
            severity_level = 'critical'
        elif severity_score >= self.severity_levels['high']:
            severity_level = 'high'
        elif severity_score >= self.severity_levels['medium']:
            severity_level = 'medium'
        else:
            severity_level = 'low'

        return {
            'severity_score': severity_score,
            'severity_level': severity_level,
            'requires_immediate_intervention': severity_level in ['critical', 'high'],
            'risk_factors': self.identify_risk_factors(features)
        }

    def identify_risk_factors(self, features: Dict) -> List[str]:
        """Identify specific risk factors present in the text"""
        risk_factors = []

        if features['suicide_keywords'] > 0:
            risk_factors.append('Suicidal ideation detected')
        if features['self_harm_keywords'] > 0:
            risk_factors.append('Self-harm indicators present')
        if features['plans_keywords'] > 0:
            risk_factors.append('Potential suicide plan mentioned')
        if features['hopelessness_keywords'] > 2:
            risk_factors.append('High level of hopelessness')
        if features['sentiment_compound'] < -0.8:
            risk_factors.append('Extremely negative emotional state')
        if features['isolation_keywords'] > 1:
            risk_factors.append('Social isolation indicators')

        return risk_factors

    def generate_intervention_response(self, severity_assessment: Dict) -> Dict:
        """Generate appropriate intervention response based on severity"""

        severity_level = severity_assessment['severity_level']

        if severity_level == 'critical':
            return {
                'response_type': 'emergency_intervention',
                'immediate_action': 'contact_emergency_services',
                'message': "I'm very concerned about what you're sharing. Your safety is the top priority right now. Please contact emergency services immediately or call the crisis hotline.",
                'resources': self.crisis_resources['suicide_hotlines'],
                'follow_up': 'human_counselor_required',
                'estimated_risk': 'imminent_danger'
            }

        elif severity_level == 'high':
            return {
                'response_type': 'urgent_support',
                'immediate_action': 'provide_crisis_resources',
                'message': "I can hear that you're going through an incredibly difficult time. You don't have to face this alone. Please consider reaching out to a crisis counselor who can provide the support you need.",
                'resources': {
                    'hotlines': self.crisis_resources['suicide_hotlines'],
                    'text_support': self.crisis_resources['text_support']
                },
                'follow_up': 'schedule_check_in',
                'estimated_risk': 'high_risk'
            }

        elif severity_level == 'medium':
            return {
                'response_type': 'supportive_guidance',
                'immediate_action': 'provide_support_resources',
                'message': "It sounds like you're dealing with some difficult feelings. Talking to someone can really help. Would you like me to share some resources for support?",
                'resources': {
                    'hotlines': self.crisis_resources['suicide_hotlines'],
                    'self_help': 'Consider speaking with a mental health professional'
                },
                'follow_up': 'monitor_conversation',
                'estimated_risk': 'moderate_risk'
            }

        else:  # low severity
            return {
                'response_type': 'gentle_support',
                'immediate_action': 'offer_resources',
                'message': "I notice you might be going through a tough time. Remember that support is available if you need it.",
                'resources': {
                    'general_support': 'Mental health resources available if needed'
                },
                'follow_up': 'continue_monitoring',
                'estimated_risk': 'low_risk'
            }

    def predict_crisis(self, text: str) -> Dict:
        """Main function to predict crisis level and provide intervention"""

        # Extract features
        features = self.extract_crisis_features(text)

        # Assess severity
        severity_assessment = self.assess_crisis_severity(features)

        # Generate intervention response
        intervention_response = self.generate_intervention_response(severity_assessment)

        # ML prediction if model is trained
        ml_prediction = None
        if self.is_trained:
            try:
                # Prepare features for ML model
                feature_vector = self.prepare_ml_features(features)
                crisis_probability = self.crisis_classifier.predict_proba([feature_vector])[0][1]
                ml_prediction = {
                    'crisis_probability': crisis_probability,
                    'crisis_predicted': crisis_probability > self.crisis_threshold
                }
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {str(e)}")

        # Compile final result
        result = {
            'text_analyzed': text,
            'timestamp': datetime.now().isoformat(),
            'crisis_detected': severity_assessment['requires_immediate_intervention'],
            'severity_assessment': severity_assessment,
            'intervention_response': intervention_response,
            'crisis_features': features,
            'analysis_method': 'rule_based'
        }

        if ml_prediction:
            result['ml_prediction'] = ml_prediction
            result['analysis_method'] = 'hybrid'

            # Ensemble decision (combine rule-based and ML)
            ensemble_score = (severity_assessment['severity_score'] + ml_prediction['crisis_probability']) / 2
            result['ensemble_crisis_score'] = ensemble_score
            result['ensemble_crisis_detected'] = ensemble_score > self.crisis_threshold

        return result

    def prepare_ml_features(self, features: Dict) -> List[float]:
        """Prepare features for ML model"""
        # Select relevant numerical features for ML model
        ml_features = [
            features['sentiment_compound'],
            features['sentiment_negative'],
            features['polarity'],
            features['suicide_keywords'],
            features['self_harm_keywords'],
            features['hopelessness_keywords'],
            features['despair_keywords'],
            features['plans_keywords'],
            features['first_person_pronouns'],
            features['negative_words'],
            features['intensity_words'],
            features['total_crisis_keywords']
        ]

        return ml_features

    def train_model(self, training_data: pd.DataFrame):
        """Train the crisis detection model"""

        if 'text' not in training_data.columns or 'crisis_detected' not in training_data.columns:
            raise ValueError("Training data must contain 'text' and 'crisis_detected' columns")

        # Extract features for all training samples
        feature_list = []
        labels = []

        for _, row in training_data.iterrows():
            features = self.extract_crisis_features(row['text'])
            ml_features = self.prepare_ml_features(features)
            feature_list.append(ml_features)
            labels.append(1 if row['crisis_detected'] else 0)

        # Train the classifier
        if len(feature_list) > 10:
            self.crisis_classifier.fit(feature_list, labels)
            self.is_trained = True
            self.logger.info(f"Crisis intervention model trained on {len(feature_list)} samples")
        else:
            self.logger.warning("Insufficient training data for crisis model")

    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        if not self.is_trained:
            return {"error": "Model not trained"}

        predictions = []
        true_labels = []

        for _, row in test_data.iterrows():
            result = self.predict_crisis(row['text'])
            predictions.append(1 if result['crisis_detected'] else 0)
            true_labels.append(1 if row['crisis_detected'] else 0)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'samples_tested': len(test_data)
        }

    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'crisis_classifier': self.crisis_classifier,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'is_trained': self.is_trained,
            'crisis_threshold': self.crisis_threshold,
            'crisis_keywords': self.crisis_keywords,
            'severity_levels': self.severity_levels,
            'crisis_resources': self.crisis_resources
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Crisis intervention model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.crisis_classifier = model_data['crisis_classifier']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.is_trained = model_data['is_trained']
        self.crisis_threshold = model_data.get('crisis_threshold', 0.8)
        self.crisis_keywords = model_data.get('crisis_keywords', self.crisis_keywords)
        self.severity_levels = model_data.get('severity_levels', self.severity_levels)
        self.crisis_resources = model_data.get('crisis_resources', self.crisis_resources)

        self.logger.info(f"Crisis intervention model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    crisis_model = CrisisInterventionModel()

    # Test with various crisis scenarios
    test_texts = [
        "I don't want to live anymore and I have a plan",  # Critical
        "I feel so hopeless and worthless",  # High
        "I'm really struggling with depression lately",  # Medium
        "Having a tough day at work"  # Low
    ]

    print("=== Crisis Intervention Model ===")
    for text in test_texts:
        result = crisis_model.predict_crisis(text)
        print(f"\nText: {text}")
        print(f"Crisis detected: {result['crisis_detected']}")
        print(f"Severity: {result['severity_assessment']['severity_level']}")
        print(f"Response: {result['intervention_response']['response_type']}")
