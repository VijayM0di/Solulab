
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging
from typing import List, Dict, Tuple
import pickle
import os
from datetime import datetime

class AbuseDetectionModel:
    """
    Advanced Abuse Language Detection Model using BERT-based transformers
    Detects multiple types of toxic content: toxic, severe_toxic, obscene, threat, insult, identity_hate
    """

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.num_labels = len(self.labels)

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load pre-trained BERT model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=self.num_labels,
                problem_type="multi_label_classification"
            )
            self.model.to(self.device)
            self.logger.info(f"Model {self.model_name} loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_data(self, texts: List[str], labels: np.ndarray = None):
        """Preprocess text data for model input"""
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Create dataset
        dataset = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }

        if labels is not None:
            dataset['labels'] = torch.FloatTensor(labels)

        return dataset

    def train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None, 
                   epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the abuse detection model"""

        if not self.model:
            self.load_model()

        # Prepare training data
        train_texts = train_data['comment_text'].tolist()
        train_labels = train_data[self.labels].values.astype(float)

        train_dataset = self.preprocess_data(train_texts, train_labels)

        # Prepare validation data if provided
        eval_dataset = None
        if val_data is not None:
            val_texts = val_data['comment_text'].tolist()
            val_labels = val_data[self.labels].values.astype(float)
            eval_dataset = self.preprocess_data(val_texts, val_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
        )

        # Custom dataset class
        class ToxicityDataset(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset['input_ids'])

            def __getitem__(self, idx):
                item = {}
                for key, val in self.dataset.items():
                    item[key] = val[idx]
                return item

        train_dataset = ToxicityDataset(train_dataset)
        eval_dataset = ToxicityDataset(eval_dataset) if eval_dataset else None

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Train the model
        self.logger.info("Starting model training...")
        trainer.train()
        self.logger.info("Training completed!")

        return trainer

    def predict(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """Predict toxicity for given texts"""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.eval()
        results = []

        with torch.no_grad():
            # Preprocess input
            dataset = self.preprocess_data(texts)

            # Get predictions
            outputs = self.model(
                input_ids=dataset['input_ids'].to(self.device),
                attention_mask=dataset['attention_mask'].to(self.device)
            )

            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(outputs.logits).cpu().numpy()

            # Process results
            for i, text in enumerate(texts):
                predictions = {}
                for j, label in enumerate(self.labels):
                    prob = float(probabilities[i][j])
                    predictions[label] = {
                        'probability': prob,
                        'prediction': prob > threshold
                    }

                # Overall toxicity score
                overall_toxicity = max([predictions[label]['probability'] for label in self.labels])

                results.append({
                    'text': text,
                    'predictions': predictions,
                    'overall_toxicity_score': overall_toxicity,
                    'is_toxic': overall_toxicity > threshold,
                    'timestamp': datetime.now().isoformat()
                })

        return results

    def evaluate_model(self, test_data: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        test_texts = test_data['comment_text'].tolist()
        test_labels = test_data[self.labels].values.astype(float)

        # Get predictions
        predictions = self.predict(test_texts)

        # Extract predicted probabilities
        pred_probs = np.array([[pred['predictions'][label]['probability'] 
                              for label in self.labels] for pred in predictions])

        pred_binary = (pred_probs > 0.5).astype(int)

        # Calculate metrics
        metrics = {}
        for i, label in enumerate(self.labels):
            accuracy = accuracy_score(test_labels[:, i], pred_binary[:, i])
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_labels[:, i], pred_binary[:, i], average='binary'
            )
            try:
                auc = roc_auc_score(test_labels[:, i], pred_probs[:, i])
            except ValueError:
                auc = 0.0  # In case of no positive samples

            metrics[label] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc
            }

        # Overall metrics
        overall_accuracy = accuracy_score(test_labels, pred_binary)
        metrics['overall'] = {
            'accuracy': overall_accuracy,
            'samples_tested': len(test_data)
        }

        return metrics

    def save_model(self, save_path: str):
        """Save the trained model"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save additional model info
        model_info = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'labels': self.labels,
            'save_timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(save_path, 'model_info.json'), 'w') as f:
            import json
            json.dump(model_info, f, indent=2)

        self.logger.info(f"Model saved to {save_path}")

    def load_saved_model(self, model_path: str):
        """Load a previously saved model"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.to(self.device)

            # Load model info if available
            info_path = os.path.join(model_path, 'model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    import json
                    model_info = json.load(f)
                    self.max_length = model_info.get('max_length', 512)
                    self.labels = model_info.get('labels', self.labels)

            self.logger.info(f"Saved model loaded from {model_path}")

        except Exception as e:
            self.logger.error(f"Error loading saved model: {str(e)}")
            raise

    def real_time_detection(self, text: str) -> Dict:
        """Real-time abuse detection for a single text"""
        results = self.predict([text])
        return results[0] if results else None

# Example usage and testing
if __name__ == "__main__":
    # Initialize model
    abuse_model = AbuseDetectionModel()

    # Example testing with sample data
    sample_texts = [
        "You are such an idiot!",
        "Thank you for your help!",
        "I hate people like you",
        "This is a great discussion"
    ]

    print("=== Abuse Detection Model ===")
    print("Model initialized successfully!")
    print("Sample texts for testing:", sample_texts)
