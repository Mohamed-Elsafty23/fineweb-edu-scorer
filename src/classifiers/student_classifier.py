"""
Student classifier for educational content scoring.
Uses embeddings + regression to mimic the Teacher model's scoring.
"""

import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from typing import Dict, Tuple
import os

from src.utils.api_client import api_client
import config


class StudentClassifier:
    """
    Lightweight student classifier trained to predict educational scores.
    Uses text embeddings as input features for regression.
    """
    
    def __init__(self, embedding_model: str = None):
        """
        Initialize the student classifier.
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.regressor = Ridge(alpha=1.0)  # Ridge regression with L2 regularization
        self.is_trained = False
    
    def get_embeddings(self, texts: list) -> np.ndarray:
        """
        Get embeddings for texts using the API.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Truncate texts to fit within embedding model's token limit (~512 tokens)
        truncated_texts = []
        for text in texts:
            # Limit to approximately 200 words (~1000 characters) to stay well under 512 token limit
            words = text.split()
            if len(words) > 200:
                text = ' '.join(words[:200])
            truncated_texts.append(text)
        
        return api_client.get_embeddings(truncated_texts, model=self.embedding_model)
    
    def train(self, texts: list, scores: list) -> Dict:
        """
        Train the student classifier on annotated data.
        
        Args:
            texts: List of text strings
            scores: List of educational scores (0-5)
            
        Returns:
            Dictionary of training metrics
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.get_embeddings(texts)
        
        print(f"Training regression model...")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Training samples: {embeddings.shape[0]}")
        
        self.regressor.fit(embeddings, scores)
        self.is_trained = True
        
        # Calculate training metrics
        predictions = self.regressor.predict(embeddings)
        metrics = self._calculate_metrics(scores, predictions)
        
        print(f"\nTraining metrics:")
        self._print_metrics(metrics)
        
        return metrics
    
    def predict(self, text: str) -> float:
        """
        Predict educational score for a single text.
        
        Args:
            text: Text string to score
            
        Returns:
            Predicted educational score (0-5, continuous)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        embedding = self.get_embeddings([text])
        prediction = self.regressor.predict(embedding)[0]
        
        # Clip to valid range [0, 5]
        return np.clip(prediction, 0, 5)
    
    def predict_batch(self, texts: list) -> np.ndarray:
        """
        Predict educational scores for multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of predicted scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        embeddings = self.get_embeddings(texts)
        predictions = self.regressor.predict(embeddings)
        
        # Clip to valid range
        return np.clip(predictions, 0, 5)
    
    def evaluate(self, texts: list, true_scores: list) -> Dict:
        """
        Evaluate the classifier on test data.
        
        Args:
            texts: List of text strings
            true_scores: List of true educational scores
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict_batch(texts)
        metrics = self._calculate_metrics(true_scores, predictions)
        
        print(f"\nEvaluation metrics:")
        self._print_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(self, true_scores: list, predictions: np.ndarray) -> Dict:
        """
        Calculate regression and classification metrics.
        
        Args:
            true_scores: True scores
            predictions: Predicted scores
            
        Returns:
            Dictionary of metrics
        """
        # Regression metrics
        mae = mean_absolute_error(true_scores, predictions)
        rmse = np.sqrt(mean_squared_error(true_scores, predictions))
        r2 = r2_score(true_scores, predictions)
        
        # Binary classification metrics (threshold at 3)
        true_binary = [1 if s >= config.EDUCATIONAL_THRESHOLD else 0 for s in true_scores]
        pred_binary = [1 if s >= config.EDUCATIONAL_THRESHOLD else 0 for s in predictions]
        
        accuracy = accuracy_score(true_binary, pred_binary)
        f1 = f1_score(true_binary, pred_binary)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def _print_metrics(self, metrics: Dict):
        """Print metrics in a formatted way."""
        print(f"  MAE:  {metrics['mae']:.3f}")
        print(f"  RMSE: {metrics['rmse']:.3f}")
        print(f"  R²:   {metrics['r2']:.3f}")
        print(f"  Binary Classification (threshold={config.EDUCATIONAL_THRESHOLD}):")
        print(f"    Accuracy: {metrics['accuracy']:.3f}")
        print(f"    F1 Score: {metrics['f1_score']:.3f}")
    
    def save(self, path: str = None):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        path = path or config.STUDENT_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'regressor': self.regressor,
            'embedding_model': self.embedding_model,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        print(f"\n✓ Model saved to {path}")
    
    @classmethod
    def load(cls, path: str = None):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded StudentClassifier instance
        """
        path = path or config.STUDENT_MODEL_PATH
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        classifier = cls(embedding_model=model_data['embedding_model'])
        classifier.regressor = model_data['regressor']
        classifier.is_trained = model_data['is_trained']
        
        print(f"✓ Model loaded from {path}")
        return classifier


def test_student_classifier():
    """Test the student classifier with dummy data."""
    print("Testing Student Classifier...")
    
    # Create dummy data
    texts = [
        "This is an educational text about science.",
        "Click here for amazing deals!",
        "Mathematics is the study of numbers and patterns."
    ]
    scores = [4, 1, 5]
    
    # Train
    classifier = StudentClassifier()
    classifier.train(texts, scores)
    
    # Predict
    test_text = "Physics is a branch of science."
    prediction = classifier.predict(test_text)
    print(f"\nTest prediction: {prediction:.2f}")
    
    print("\n✓ Student classifier test passed!")


if __name__ == "__main__":
    test_student_classifier()

