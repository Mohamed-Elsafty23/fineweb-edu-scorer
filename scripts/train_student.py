import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config
from src.classifiers.student_classifier import StudentClassifier


def load_training_data(path: str = None) -> tuple:
    path = path or config.TRAINING_DATA_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training data not found at {path}\n"
            f"Please run 'python scripts/create_training_data.py' first."
        )
    
    print(f"Loading training data from {path}...")
    df = pd.read_csv(path)
    
    df = df[df['score'].notna()].copy()
    
    print(f"Loaded {len(df)} annotated samples")
    print(f"\nDataset statistics:")
    print(f"  Score range: {df['score'].min():.1f} - {df['score'].max():.1f}")
    print(f"  Mean score: {df['score'].mean():.2f}")
    print(f"  Score distribution:")
    print(df['score'].value_counts().sort_index())
    
    texts = df['text'].tolist()
    scores = df['score'].tolist()
    
    return texts, scores


def train_and_evaluate(texts: list, scores: list, test_size: float = 0.2):
    print("\n" + "=" * 60)
    print("Training Student Classifier")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, scores, test_size=test_size, random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    print(f"\nInitializing Student classifier...")
    print(f"  Embedding model: {config.EMBEDDING_MODEL}")
    
    classifier = StudentClassifier()
    
    print("\n--- Training Phase ---")
    train_metrics = classifier.train(X_train, y_train)
    
    print("\n--- Evaluation Phase ---")
    test_metrics = classifier.evaluate(X_test, y_test)
    
    print("\n--- Sample Predictions ---")
    num_samples = min(5, len(X_test))
    for i in range(num_samples):
        pred = classifier.predict(X_test[i])
        print(f"\nSample {i+1}:")
        print(f"  True score: {y_test[i]}")
        print(f"  Predicted:  {pred:.2f}")
        print(f"  Text preview: {X_test[i][:100]}...")
    
    return classifier


def main():
    print("=" * 60)
    print("FineWeb-Edu Student Classifier Training")
    print("=" * 60)
    
    try:
        texts, scores = load_training_data()
        
        if len(texts) < 10:
            print(f"\nWarning: Only {len(texts)} samples available. Consider generating more training data.")
            print("Continuing with available data...")
        
        classifier = train_and_evaluate(texts, scores)
        
        print("\n--- Saving Model ---")
        classifier.save()
        
        print("\n" + "=" * 60)
        print("Student classifier training complete!")
        print("=" * 60)
        print(f"\nModel saved to: {config.STUDENT_MODEL_PATH}")
        print(f"\nNext step: Run 'streamlit run app.py' to launch the interactive UI")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
