
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class SampleDataGenerator:
    '''Generate sample data for testing AI Safety Models'''

    def __init__(self):
        # Sample toxic comments for abuse detection
        self.toxic_samples = [
            "You're such an idiot, go kill yourself",
            "I hate people like you, you should die",
            "Stupid moron, you're worthless",
            "You're disgusting and pathetic",
            "I hope you get what you deserve, loser"
        ]

        # Sample escalation conversation patterns
        self.escalation_patterns = [
            ["I'm a bit frustrated", "This is really annoying", "I'm getting angry", "I'm furious!", "I hate this!"],
            ["Can you help me?", "This isn't working", "Why is this so hard?", "This is stupid!", "I give up!"],
            ["I disagree", "That's wrong", "You don't understand", "You're being unreasonable", "This is ridiculous!"]
        ]

        # Sample crisis intervention messages
        self.crisis_samples = [
            "I don't see the point in living anymore",
            "I'm thinking about ending it all",
            "Nobody would miss me if I was gone",
            "I have a plan to hurt myself",
            "Life is too painful to continue",
            "I just want the pain to stop forever"
        ]

        # Sample age-inappropriate content
        self.inappropriate_content = [
            {"text": "Violent movie scene description", "age_restriction": 18, "category": "violence"},
            {"text": "Mature themes discussion", "age_restriction": 16, "category": "mature_themes"},
            {"text": "Drug usage tutorial", "age_restriction": 21, "category": "drugs"},
            {"text": "Adult content reference", "age_restriction": 18, "category": "sexual"}
        ]

    def generate_abuse_detection_data(self, n_samples=1000):
        '''Generate labeled data for abuse detection model'''
        data = []

        # Generate toxic samples
        for _ in range(n_samples // 2):
            text = random.choice(self.toxic_samples)
            # Add some variation
            text += f" {random.choice(['', '!!!', '...', '???'])}"

            labels = {
                'toxic': random.choice([0, 1]),
                'severe_toxic': random.choice([0, 1]) if random.random() > 0.8 else 0,
                'obscene': random.choice([0, 1]) if random.random() > 0.7 else 0,
                'threat': random.choice([0, 1]) if random.random() > 0.9 else 0,
                'insult': random.choice([0, 1]) if random.random() > 0.6 else 0,
                'identity_hate': random.choice([0, 1]) if random.random() > 0.85 else 0
            }

            data.append({
                'id': f"toxic_{len(data)}",
                'comment_text': text,
                **labels
            })

        # Generate non-toxic samples
        non_toxic_samples = [
            "Thank you for your help!",
            "I appreciate your feedback",
            "This is a great discussion",
            "I learned something new today",
            "Have a wonderful day!"
        ]

        for _ in range(n_samples // 2):
            text = random.choice(non_toxic_samples)

            labels = {
                'toxic': 0,
                'severe_toxic': 0,
                'obscene': 0,
                'threat': 0,
                'insult': 0,
                'identity_hate': 0
            }

            data.append({
                'id': f"non_toxic_{len(data)}",
                'comment_text': text,
                **labels
            })

        return pd.DataFrame(data)

    def generate_conversation_data(self, n_conversations=100):
        '''Generate conversation data for escalation detection'''
        conversations = []

        for i in range(n_conversations):
            pattern = random.choice(self.escalation_patterns)
            conversation = {
                'conversation_id': f"conv_{i}",
                'messages': [],
                'escalation_detected': len(pattern) > 3,
                'escalation_level': len(pattern) / 5.0
            }

            for j, message in enumerate(pattern):
                conversation['messages'].append({
                    'message_id': f"msg_{j}",
                    'timestamp': datetime.now() - timedelta(minutes=j*2),
                    'text': message,
                    'sentiment_score': (j + 1) / len(pattern),  # Escalating sentiment
                    'anger_level': j / (len(pattern) - 1) if len(pattern) > 1 else 0
                })

            conversations.append(conversation)

        return conversations

    def generate_crisis_data(self, n_samples=500):
        '''Generate data for crisis intervention detection'''
        data = []

        # Crisis samples
        for i in range(n_samples // 2):
            text = random.choice(self.crisis_samples)
            severity = random.choice(['medium', 'high', 'critical'])

            data.append({
                'id': f"crisis_{i}",
                'text': text,
                'crisis_detected': 1,
                'severity_level': severity,
                'requires_intervention': 1 if severity in ['high', 'critical'] else 0,
                'timestamp': datetime.now() - timedelta(hours=random.randint(0, 72))
            })

        # Non-crisis samples
        normal_samples = [
            "I had a tough day at work",
            "Feeling a bit sad today",
            "Things could be better",
            "I'm going through a difficult time",
            "Need some advice on my situation"
        ]

        for i in range(n_samples // 2):
            text = random.choice(normal_samples)

            data.append({
                'id': f"normal_{i}",
                'text': text,
                'crisis_detected': 0,
                'severity_level': 'low',
                'requires_intervention': 0,
                'timestamp': datetime.now() - timedelta(hours=random.randint(0, 72))
            })

        return pd.DataFrame(data)

    def generate_content_filtering_data(self, n_samples=300):
        '''Generate data for age-appropriate content filtering'''
        data = []

        for i in range(n_samples):
            content = random.choice(self.inappropriate_content)
            user_age = random.randint(8, 25)

            data.append({
                'content_id': f"content_{i}",
                'text': content['text'],
                'category': content['category'],
                'age_restriction': content['age_restriction'],
                'user_age': user_age,
                'should_filter': 1 if user_age < content['age_restriction'] else 0,
                'timestamp': datetime.now() - timedelta(days=random.randint(0, 30))
            })

        return pd.DataFrame(data)

# Generate and save sample datasets
if __name__ == "__main__":
    generator = SampleDataGenerator()

    # Generate datasets
    abuse_data = generator.generate_abuse_detection_data(1000)
    conversation_data = generator.generate_conversation_data(100)
    crisis_data = generator.generate_crisis_data(500)
    content_data = generator.generate_content_filtering_data(300)

    print("Sample datasets generated successfully!")
    print(f"Abuse detection data: {len(abuse_data)} samples")
    print(f"Crisis intervention data: {len(crisis_data)} samples")
    print(f"Content filtering data: {len(content_data)} samples")
    print(f"Conversation data: {len(conversation_data)} conversations")
