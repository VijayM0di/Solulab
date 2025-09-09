
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

class ContentFilteringModel:
    """
    Age-Appropriate Content Filtering System
    Filters content based on user age profiles and content categories
    Ensures children and teens only access appropriate content
    """

    def __init__(self):
        # Age categories and restrictions
        self.age_categories = {
            'child': {'min_age': 0, 'max_age': 12},
            'teen': {'min_age': 13, 'max_age': 17},
            'young_adult': {'min_age': 18, 'max_age': 25},
            'adult': {'min_age': 26, 'max_age': 100}
        }

        # Content categories with age restrictions
        self.content_restrictions = {
            'violence': {
                'mild': 13,      # Cartoon violence, mild action
                'moderate': 16,  # Realistic violence, fighting
                'severe': 18,    # Graphic violence, gore
                'extreme': 21    # Extreme graphic content
            },
            'sexual': {
                'mild': 16,      # Mild romantic content
                'moderate': 18,  # Sexual themes, suggestive content
                'explicit': 21   # Explicit sexual content
            },
            'language': {
                'mild': 10,      # Mild profanity
                'moderate': 13,  # Strong language
                'severe': 16,    # Very strong language
                'extreme': 18    # Extremely offensive language
            },
            'drugs': {
                'reference': 13, # Drug references
                'use': 16,       # Drug use depiction
                'detailed': 18,  # Detailed drug use
                'instructional': 21  # Drug manufacturing/dealing
            },
            'mature_themes': {
                'mild': 13,      # Coming of age themes
                'moderate': 16,  # Complex social issues
                'severe': 18,    # Dark psychological themes
                'adult': 21      # Adult life challenges
            }
        }

        # Keyword dictionaries for content detection
        self.content_keywords = {
            'violence': {
                'mild': ['fight', 'battle', 'combat', 'action', 'adventure', 'superhero'],
                'moderate': ['violence', 'violent', 'attack', 'weapon', 'gun', 'sword', 'war'],
                'severe': ['blood', 'gore', 'brutal', 'murder', 'kill', 'death', 'torture'],
                'extreme': ['massacre', 'slaughter', 'dismember', 'mutilation', 'graphic violence']
            },
            'sexual': {
                'mild': ['romance', 'kiss', 'love', 'dating', 'relationship'],
                'moderate': ['sexual', 'intimate', 'seductive', 'provocative', 'suggestive'],
                'explicit': ['sex', 'porn', 'nude', 'naked', 'explicit', 'erotic']
            },
            'language': {
                'mild': ['damn', 'hell', 'crap', 'stupid'],
                'moderate': ['ass', 'bitch', 'shit', 'bastard'],
                'severe': ['fuck', 'fucking', 'motherfucker'],
                'extreme': ['extreme slurs', 'hate speech']  # Placeholder - actual implementation would be more careful
            },
            'drugs': {
                'reference': ['alcohol', 'beer', 'wine', 'cigarette', 'smoking'],
                'use': ['drunk', 'high', 'marijuana', 'weed', 'drug use'],
                'detailed': ['cocaine', 'heroin', 'meth', 'addiction'],
                'instructional': ['drug dealing', 'manufacturing', 'trafficking']
            },
            'mature_themes': {
                'mild': ['growing up', 'puberty', 'identity', 'bullying'],
                'moderate': ['depression', 'anxiety', 'social issues', 'discrimination'],
                'severe': ['suicide', 'self-harm', 'abuse', 'trauma'],
                'adult': ['financial stress', 'divorce', 'career pressure', 'existential crisis']
            }
        }

        # Machine learning components
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.content_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def determine_age_category(self, age: int) -> str:
        """Determine age category based on user age"""
        for category, age_range in self.age_categories.items():
            if age_range['min_age'] <= age <= age_range['max_age']:
                return category
        return 'adult'  # Default for ages outside defined ranges

    def extract_content_features(self, text: str) -> Dict:
        """Extract features for content analysis"""
        features = {}
        text_lower = text.lower()

        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))

        # Content category analysis
        for category, intensity_keywords in self.content_keywords.items():
            category_score = 0
            max_intensity = 'none'

            for intensity, keywords in intensity_keywords.items():
                keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
                features[f'{category}_{intensity}_keywords'] = keyword_count

                if keyword_count > 0:
                    category_score = max(category_score, list(intensity_keywords.keys()).index(intensity) + 1)
                    max_intensity = intensity

            features[f'{category}_score'] = category_score
            features[f'{category}_max_intensity'] = max_intensity

        # Overall content analysis
        features['total_inappropriate_keywords'] = sum(
            features.get(f'{cat}_{int}_keywords', 0)
            for cat in self.content_keywords.keys()
            for int in self.content_keywords[cat].keys()
        )

        # Sentiment and tone
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity

        # Linguistic patterns
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')

        return features

    def assess_content_appropriateness(self, text: str, user_age: int) -> Dict:
        """Assess if content is appropriate for user's age"""

        features = self.extract_content_features(text)
        age_category = self.determine_age_category(user_age)
        max_required_age = 0
        # Rule-based assessment
        content_issues = []
        should_filter = False
        

        for category, intensities in self.content_restrictions.items():
            category_score = features.get(f'{category}_score', 0)
            max_intensity = features.get(f'{category}_max_intensity', 'none')

            if category_score > 0 and max_intensity in intensities:
                required_age = intensities[max_intensity]
                max_required_age = max(max_required_age, required_age)

                if user_age < required_age:
                    should_filter = True
                    content_issues.append({
                        'category': category,
                        'intensity': max_intensity,
                        'required_age': required_age,
                        'user_age': user_age
                    })

        # Additional checks for edge cases
        if features['total_inappropriate_keywords'] > 5:  # High concentration of inappropriate content
            should_filter = True
            content_issues.append({
                'category': 'general',
                'reason': 'high_concentration_inappropriate_content',
                'keyword_count': features['total_inappropriate_keywords']
            })

        # Parental guidance suggestions
        guidance_level = self.determine_guidance_level(max_required_age, user_age)

        return {
            'should_filter': bool(should_filter),
            'user_age': int(user_age),
            'user_age_category': age_category,
            'content_issues': content_issues,
            'max_required_age': int(max_required_age),
            'guidance_level': guidance_level,
            'status': 'completed',
            'content_features': features,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def determine_guidance_level(self, content_age_requirement: int, user_age: int) -> str:
        """Determine what level of parental guidance is recommended"""

        if content_age_requirement == 0:
            return 'none'
        elif user_age >= content_age_requirement:
            return 'none'
        elif content_age_requirement - user_age <= 2:
            return 'mild'
        elif content_age_requirement - user_age <= 5:
            return 'moderate'
        else:
            return 'strict'

    def generate_filtering_response(self, assessment: Dict) -> Dict:
        """Generate appropriate response for content filtering"""

        if not assessment['should_filter']:
            return {
                'action': 'allow',
                'message': 'Content approved for user age group',
                'restrictions': None
            }

        # Determine filtering action
        if assessment['max_required_age'] - assessment['user_age'] > 5:
            action = 'block'
            message = "This content is not appropriate for your age group and has been blocked."
        elif assessment['max_required_age'] - assessment['user_age'] > 2:
            action = 'require_parental_approval'
            message = "This content requires parental approval before viewing."
        else:
            action = 'show_warning'
            message = "This content may contain material that some viewers find inappropriate."

        # Generate specific guidance
        guidance = self.generate_parental_guidance(assessment)

        return {
            'action': action,
            'message': message,
            'content_issues': assessment['content_issues'],
            'required_age': assessment['max_required_age'],
            'guidance_level': assessment['guidance_level'],
            'parental_guidance': guidance,
            'alternative_suggestions': self.get_alternative_content_suggestions(assessment)
        }

    def generate_parental_guidance(self, assessment: Dict) -> Dict:
        """Generate parental guidance recommendations"""

        guidance = {
            'supervision_recommended': assessment['guidance_level'] in ['moderate', 'strict'],
            'discussion_topics': [],
            'watch_together': assessment['guidance_level'] == 'strict'
        }

        for issue in assessment['content_issues']:
            category = issue['category']

            if category == 'violence':
                guidance['discussion_topics'].append(
                    "Discuss the difference between real and fictional violence"
                )
            elif category == 'sexual':
                guidance['discussion_topics'].append(
                    "Age-appropriate discussion about relationships and personal boundaries"
                )
            elif category == 'mature_themes':
                guidance['discussion_topics'].append(
                    "Help process complex themes and emotions presented"
                )
            elif category == 'drugs':
                guidance['discussion_topics'].append(
                    "Discuss the risks and consequences of substance use"
                )

        return guidance

    def get_alternative_content_suggestions(self, assessment: Dict) -> List[str]:
        """Suggest alternative age-appropriate content"""

        age_category = assessment['user_age_category']

        alternatives = {
            'child': [
                "Educational cartoons and animated shows",
                "Nature documentaries for children",
                "Interactive learning games",
                "Age-appropriate story books"
            ],
            'teen': [
                "Coming-of-age stories appropriate for teens",
                "Educational content about science and history",
                "Age-appropriate documentaries",
                "Creative and artistic content"
            ],
            'young_adult': [
                "Young adult literature and films",
                "Career and educational content",
                "Social awareness documentaries",
                "Creative and cultural content"
            ]
        }

        return alternatives.get(age_category, ["General audience content"])

    def batch_filter_content(self, content_list: List[Dict]) -> List[Dict]:
        """Filter multiple pieces of content for batch processing"""

        results = []

        for content_item in content_list:
            text = content_item.get('text', '')
            user_age = content_item.get('user_age', 18)
            content_id = content_item.get('id', 'unknown')

            assessment = self.assess_content_appropriateness(text, user_age)
            filtering_response = self.generate_filtering_response(assessment)

            result = {
                'content_id': content_id,
                'user_age': user_age,
                'assessment': assessment,
                'filtering_response': filtering_response,
                'timestamp': datetime.now().isoformat()
            }

            results.append(result)

        return results

    def train_content_classifier(self, training_data: pd.DataFrame):
        """Train machine learning model for content classification"""

        if 'text' not in training_data.columns or 'should_filter' not in training_data.columns:
            raise ValueError("Training data must contain 'text' and 'should_filter' columns")

        # Extract features for training
        feature_list = []
        labels = []

        for _, row in training_data.iterrows():
            features = self.extract_content_features(row['text'])

            # Create feature vector for ML
            feature_vector = [
                features['text_length'],
                features['word_count'],
                features['violence_score'],
                features['sexual_score'],
                features['language_score'],
                features['drugs_score'],
                features['mature_themes_score'],
                features['total_inappropriate_keywords'],
                features['caps_ratio']
            ]

            feature_list.append(feature_vector)
            labels.append(1 if row['should_filter'] else 0)

        # Train classifier
        if len(feature_list) > 10:
            self.content_classifier.fit(feature_list, labels)
            self.is_trained = True
            self.logger.info(f"Content filtering model trained on {len(feature_list)} samples")
        else:
            self.logger.warning("Insufficient training data for content filtering model")

    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'content_classifier': self.content_classifier,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'is_trained': self.is_trained,
            'age_categories': self.age_categories,
            'content_restrictions': self.content_restrictions,
            'content_keywords': self.content_keywords
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Content filtering model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.content_classifier = model_data['content_classifier']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.is_trained = model_data['is_trained']
        self.age_categories = model_data.get('age_categories', self.age_categories)
        self.content_restrictions = model_data.get('content_restrictions', self.content_restrictions)
        self.content_keywords = model_data.get('content_keywords', self.content_keywords)

        self.logger.info(f"Content filtering model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    content_filter = ContentFilteringModel()

    # Test content filtering
    test_content = [
        {"id": "content_1", "text": "This movie contains graphic violence and strong language", "user_age": 14},
        {"id": "content_2", "text": "A beautiful story about friendship and growing up", "user_age": 10},
        {"id": "content_3", "text": "Educational documentary about marine biology", "user_age": 8},
        {"id": "content_4", "text": "Adult content with explicit sexual themes", "user_age": 16}
    ]

    print("=== Content Filtering Model ===")
    results = content_filter.batch_filter_content(test_content)

    for result in results:
        print(f"\nContent ID: {result['content_id']}")
        print(f"User Age: {result['user_age']}")
        print(f"Should Filter: {result['assessment']['should_filter']}")
        print(f"Action: {result['filtering_response']['action']}")
        print(f"Required Age: {result['assessment']['max_required_age']}")
