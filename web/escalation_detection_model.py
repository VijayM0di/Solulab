
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import pickle

class EscalationPatternDetector:
    """
    Advanced Escalation Pattern Recognition System
    Detects when conversations are becoming emotionally dangerous through:
    - Sentiment progression analysis
    - Linguistic pattern recognition
    - Temporal analysis of message frequency
    - Aggression level detection
    """

    def __init__(self, window_size: int = 5, escalation_threshold: float = 0.8):
        self.window_size = window_size  # Number of messages to analyze
        self.escalation_threshold = escalation_threshold
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.escalation_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

        # Escalation indicators
        self.escalation_keywords = {
            'anger': ['angry', 'furious', 'mad', 'hate', 'rage', 'frustrated', 'pissed'],
            'aggression': ['fight', 'attack', 'destroy', 'kill', 'hurt', 'violence', 'aggressive'],
            'profanity': ['damn', 'hell', 'stupid', 'idiot', 'moron', 'bastard', 'asshole'],
            'threats': ['threat', 'revenge', 'payback', 'regret', 'sorry', 'consequences'],
            'intensity': ['extremely', 'absolutely', 'totally', 'completely', 'definitely']
        }

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_linguistic_features(self, message: str) -> Dict:
        """Extract linguistic features from a message"""
        features = {}

        # Basic features
        features['message_length'] = len(message)
        features['word_count'] = len(message.split())
        features['char_count'] = len(message)
        features['exclamation_count'] = message.count('!')
        features['question_count'] = message.count('?')
        features['caps_ratio'] = sum(1 for c in message if c.isupper()) / len(message) if message else 0

        # Sentiment analysis
        sentiment = self.sentiment_analyzer.polarity_scores(message)
        features['sentiment_compound'] = sentiment['compound']
        features['sentiment_positive'] = sentiment['pos']
        features['sentiment_negative'] = sentiment['neg']
        features['sentiment_neutral'] = sentiment['neu']

        # TextBlob sentiment (alternative)
        blob = TextBlob(message)
        features['textblob_polarity'] = blob.sentiment.polarity
        features['textblob_subjectivity'] = blob.sentiment.subjectivity

        # Keyword counting
        message_lower = message.lower()
        for category, keywords in self.escalation_keywords.items():
            features[f'{category}_count'] = sum(1 for keyword in keywords if keyword in message_lower)

        # Repetition patterns
        words = message_lower.split()
        if words:
            features['word_repetition'] = max([words.count(word) for word in set(words)])
        else:
            features['word_repetition'] = 0

        # Punctuation patterns
        features['multiple_exclamations'] = len(re.findall(r'!{2,}', message))
        features['multiple_questions'] = len(re.findall(r'\?{2,}', message))
        features['ellipsis_count'] = message.count('...')

        return features

    def analyze_conversation_sequence(self, messages: List[Dict]) -> Dict:
        """Analyze a sequence of messages for escalation patterns"""
        if len(messages) < 2:
            return {'escalation_detected': False, 'escalation_score': 0.0, 'pattern': 'insufficient_data'}

        # Extract features for each message
        message_features = []
        timestamps = []

        for msg in messages[-self.window_size:]:  # Analyze last N messages
            features = self.extract_linguistic_features(msg['text'])
            features['timestamp'] = msg.get('timestamp', datetime.now())
            message_features.append(features)
            timestamps.append(msg.get('timestamp', datetime.now()))

        # Analyze temporal patterns
        temporal_analysis = self.analyze_temporal_patterns(timestamps)

        # Analyze sentiment progression
        sentiment_progression = self.analyze_sentiment_progression(message_features)

        # Analyze linguistic escalation
        linguistic_escalation = self.analyze_linguistic_escalation(message_features)

        # Calculate overall escalation score
        escalation_score = self.calculate_escalation_score(
            temporal_analysis, sentiment_progression, linguistic_escalation
        )

        return {
            'escalation_detected': escalation_score > self.escalation_threshold,
            'escalation_score': escalation_score,
            'temporal_analysis': temporal_analysis,
            'sentiment_progression': sentiment_progression,
            'linguistic_escalation': linguistic_escalation,
            'recommendation': self.get_intervention_recommendation(escalation_score),
            'analysis_timestamp': datetime.now().isoformat()
        }

    def analyze_temporal_patterns(self, timestamps: List[datetime]) -> Dict:
        """Analyze temporal patterns in message frequency"""
        if len(timestamps) < 2:
            return {'frequency_increase': 0.0, 'pattern': 'insufficient_data'}

        # Calculate time intervals between messages
        intervals = []
        for i in range(1, len(timestamps)):
            if isinstance(timestamps[i], str):
                timestamps[i] = datetime.fromisoformat(timestamps[i])
            if isinstance(timestamps[i-1], str):
                timestamps[i-1] = datetime.fromisoformat(timestamps[i-1])

            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)

        # Analyze frequency changes
        if len(intervals) > 1:
            early_intervals = intervals[:len(intervals)//2]
            later_intervals = intervals[len(intervals)//2:]

            avg_early = np.mean(early_intervals) if early_intervals else float('inf')
            avg_later = np.mean(later_intervals) if later_intervals else float('inf')

            # Frequency increase (lower intervals = higher frequency)
            frequency_increase = max(0, (avg_early - avg_later) / avg_early) if avg_early > 0 else 0
        else:
            frequency_increase = 0.0

        return {
            'frequency_increase': frequency_increase,
            'message_intervals': intervals,
            'pattern': 'increasing_frequency' if frequency_increase > 0.3 else 'stable'
        }

    def analyze_sentiment_progression(self, message_features: List[Dict]) -> Dict:
        """Analyze how sentiment changes over the conversation"""
        sentiments = [feat['sentiment_compound'] for feat in message_features]
        negative_sentiments = [feat['sentiment_negative'] for feat in message_features]

        if len(sentiments) < 2:
            return {'sentiment_decline': 0.0, 'pattern': 'insufficient_data'}

        # Calculate sentiment trend
        sentiment_trend = np.polyfit(range(len(sentiments)), sentiments, 1)[0]
        negative_trend = np.polyfit(range(len(negative_sentiments)), negative_sentiments, 1)[0]

        # Calculate volatility
        sentiment_volatility = np.std(sentiments) if len(sentiments) > 1 else 0

        return {
            'sentiment_trend': float(sentiment_trend),
            'negative_trend': float(negative_trend),
            'sentiment_volatility': float(sentiment_volatility),
            'current_sentiment': sentiments[-1],
            'sentiment_decline': max(0, -sentiment_trend),  # Positive value for declining sentiment
            'pattern': 'declining' if sentiment_trend < -0.1 else 'stable'
        }

    def analyze_linguistic_escalation(self, message_features: List[Dict]) -> Dict:
        """Analyze linguistic indicators of escalation"""

        # Aggregate features across messages
        caps_ratios = [feat['caps_ratio'] for feat in message_features]
        exclamation_counts = [feat['exclamation_count'] for feat in message_features]
        anger_counts = [feat['anger_count'] for feat in message_features]
        aggression_counts = [feat['aggression_count'] for feat in message_features]

        # Calculate escalation indicators
        caps_increase = np.mean(caps_ratios[-2:]) - np.mean(caps_ratios[:2]) if len(caps_ratios) >= 2 else 0
        exclamation_increase = np.mean(exclamation_counts[-2:]) - np.mean(exclamation_counts[:2]) if len(exclamation_counts) >= 2 else 0
        anger_progression = np.sum(anger_counts[-3:]) > np.sum(anger_counts[:3]) if len(anger_counts) >= 3 else False

        return {
            'caps_increase': max(0, caps_increase),
            'exclamation_increase': max(0, exclamation_increase),
            'anger_progression': anger_progression,
            'current_anger_level': anger_counts[-1] if anger_counts else 0,
            'current_aggression_level': aggression_counts[-1] if aggression_counts else 0,
            'linguistic_intensity': np.mean([caps_ratios[-1], exclamation_counts[-1]/5.0]) if message_features else 0
        }

    def calculate_escalation_score(self, temporal: Dict, sentiment: Dict, linguistic: Dict) -> float:
        """Calculate overall escalation score"""
        score_components = {
            'temporal': temporal['frequency_increase'] * 0.2,
            'sentiment': sentiment['sentiment_decline'] * 0.4,
            'negative_sentiment': abs(sentiment['current_sentiment']) * 0.3 if sentiment['current_sentiment'] < 0 else 0,
            'linguistic': linguistic['linguistic_intensity'] * 0.3,
            'anger_progression': 0.4 if linguistic['anger_progression'] else 0,
            'volatility': min(sentiment['sentiment_volatility'], 1.0) * 0.2
        }

        # Weighted sum with normalization
        escalation_score = sum(score_components.values()) / len(score_components)
        return min(1.0, escalation_score)  # Cap at 1.0

    def get_intervention_recommendation(self, escalation_score: float) -> Dict:
        """Get intervention recommendations based on escalation score"""
        if escalation_score >= 0.9:
            return {
                'level': 'critical',
                'action': 'immediate_intervention',
                'message': 'Immediate human moderator intervention required',
                'suggested_response': 'I notice this conversation is getting heated. Let me connect you with a human agent who can better assist you.'
            }
        elif escalation_score >= 0.7:
            return {
                'level': 'high',
                'action': 'de_escalation',
                'message': 'Active de-escalation needed',
                'suggested_response': 'I understand youre frustrated. Lets take a step back and work together to resolve this issue.'
            }
        elif escalation_score >= 0.5:
            return {
                'level': 'medium',
                'action': 'monitor_closely',
                'message': 'Monitor conversation closely',
                'suggested_response': 'I want to make sure we address your concerns properly. How can I help improve this situation?'
            }
        else:
            return {
                'level': 'low',
                'action': 'continue',
                'message': 'Conversation within normal parameters',
                'suggested_response': None
            }

    def train_escalation_classifier(self, training_conversations: List[Dict]):
        """Train the escalation detection classifier"""

        features_list = []
        labels = []

        for conversation in training_conversations:
            messages = conversation['messages']
            escalation_label = conversation.get('escalation_detected', False)

            # Extract features from conversation
            conversation_features = []
            for message in messages:
                msg_features = self.extract_linguistic_features(message['text'])
                conversation_features.append(msg_features)

            # Aggregate features for the entire conversation
            if conversation_features:
                aggregated_features = self.aggregate_conversation_features(conversation_features)
                features_list.append(aggregated_features)
                labels.append(1 if escalation_label else 0)

        # Train classifier if we have enough data
        if len(features_list) > 10:
            feature_df = pd.DataFrame(features_list)
            self.escalation_classifier.fit(feature_df, labels)
            self.is_trained = True
            self.logger.info(f"Escalation classifier trained on {len(features_list)} conversations")
        else:
            self.logger.warning("Insufficient training data for escalation classifier")

    def aggregate_conversation_features(self, conversation_features: List[Dict]) -> Dict:
        """Aggregate features across all messages in a conversation"""
        if not conversation_features:
            return {}

        aggregated = {}

        # Get all feature keys
        feature_keys = conversation_features[0].keys()

        for key in feature_keys:
            if key in ['timestamp']:
                continue

            values = [feat[key] for feat in conversation_features if key in feat]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_max'] = np.max(values)
                aggregated[f'{key}_std'] = np.std(values) if len(values) > 1 else 0
                aggregated[f'{key}_trend'] = np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0

        return aggregated

    def predict_escalation_risk(self, messages: List[Dict]) -> Dict:
        """Predict escalation risk for a conversation"""

        # Rule-based analysis (always available)
        rule_based_result = self.analyze_conversation_sequence(messages)

        # ML-based prediction (if trained)
        ml_prediction = None
        if self.is_trained and len(messages) > 0:
            try:
                conversation_features = []
                for message in messages:
                    msg_features = self.extract_linguistic_features(message['text'])
                    conversation_features.append(msg_features)

                aggregated_features = self.aggregate_conversation_features(conversation_features)

                if aggregated_features:
                    feature_df = pd.DataFrame([aggregated_features])
                    ml_probability = self.escalation_classifier.predict_proba(feature_df)[0][1]
                    ml_prediction = {
                        'escalation_probability': ml_probability,
                        'escalation_predicted': ml_probability > 0.5
                    }
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {str(e)}")

        # Combine rule-based and ML results
        result = rule_based_result
        if ml_prediction:
            result['ml_prediction'] = ml_prediction
            # Ensemble prediction (average of rule-based score and ML probability)
            ensemble_score = (rule_based_result['escalation_score'] + ml_prediction['escalation_probability']) / 2
            result['ensemble_escalation_score'] = ensemble_score
            result['ensemble_escalation_detected'] = ensemble_score > self.escalation_threshold

        return result

    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'escalation_classifier': self.escalation_classifier,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'is_trained': self.is_trained,
            'window_size': self.window_size,
            'escalation_threshold': self.escalation_threshold,
            'escalation_keywords': self.escalation_keywords
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Escalation model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.escalation_classifier = model_data['escalation_classifier']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.is_trained = model_data['is_trained']
        self.window_size = model_data.get('window_size', 5)
        self.escalation_threshold = model_data.get('escalation_threshold', 0.8)
        self.escalation_keywords = model_data.get('escalation_keywords', self.escalation_keywords)

        self.logger.info(f"Escalation model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    detector = EscalationPatternDetector()

    # Sample conversation with escalation
    sample_conversation = [
        {'text': 'Hi, I need help with my account', 'timestamp': datetime.now() - timedelta(minutes=10)},
        {'text': 'This is taking too long', 'timestamp': datetime.now() - timedelta(minutes=8)},
        {'text': 'I am getting frustrated with this', 'timestamp': datetime.now() - timedelta(minutes=5)},
        {'text': 'This is absolutely ridiculous!!!', 'timestamp': datetime.now() - timedelta(minutes=2)},
        {'text': 'I HATE this system!', 'timestamp': datetime.now()}
    ]

    result = detector.analyze_conversation_sequence(sample_conversation)
    print("=== Escalation Pattern Detection ===")
    print(f"Escalation detected: {result['escalation_detected']}")
    print(f"Escalation score: {result['escalation_score']:.3f}")
    print(f"Recommendation: {result['recommendation']['level']}")
