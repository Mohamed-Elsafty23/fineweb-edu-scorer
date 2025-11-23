"""
Generate synthetic training dataset by annotating FineWeb-Edu samples with Teacher LLM.
This creates the "gold standard" dataset for training the Student classifier.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import config
from src.annotators.teacher_annotator import TeacherAnnotator


def sample_diverse_texts(dataset_name: str = "HuggingFaceFW/fineweb-edu", 
                         num_samples: int = 100,
                         min_length: int = 500,
                         max_length: int = 10000) -> list:
    """
    Sample diverse texts from the FineWeb-Edu dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        num_samples: Number of samples to extract
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
        
    Returns:
        List of sampled texts
    """
    print(f"Loading dataset: {dataset_name}")
    print("This may take a moment as we stream from HuggingFace...")
    
    # Use streaming to avoid downloading the entire dataset
    dataset = load_dataset(
        dataset_name, 
        name="sample-10BT",  # Use the 10BT sample subset for efficiency
        split="train",
        streaming=True
    )
    
    sampled_texts = []
    seen_texts = set()  # Avoid duplicates
    
    print(f"Sampling {num_samples} diverse texts...")
    
    for sample in tqdm(dataset, total=num_samples * 3):  # Iterate through more to find good samples
        if len(sampled_texts) >= num_samples:
            break
        
        text = sample['text']
        text_length = len(text)
        
        # Filter by length
        if text_length < min_length or text_length > max_length:
            continue
        
        # Avoid duplicates (check first 100 chars)
        text_preview = text[:100]
        if text_preview in seen_texts:
            continue
        
        seen_texts.add(text_preview)
        sampled_texts.append(text)
    
    print(f"Sampled {len(sampled_texts)} texts")
    return sampled_texts


def annotate_with_teacher(texts: list, output_path: str = None) -> pd.DataFrame:
    """
    Annotate texts with Teacher model scores.
    
    Args:
        texts: List of text strings
        output_path: Path to save the annotated dataset
        
    Returns:
        DataFrame with columns: text, score, reasoning, decision
    """
    print("\nAnnotating texts with Teacher LLM...")
    print("This will take several minutes as each text is sent to the API.")
    
    annotator = TeacherAnnotator()
    
    data = []
    for i, text in enumerate(tqdm(texts, desc="Annotating")):
        print(f"\nProcessing text {i+1}/{len(texts)}")
        
        result = annotator.get_educational_score(text)
        
        data.append({
            'text': text,
            'score': result['score'],
            'reasoning': result['reasoning'],
            'decision': result['decision'],
            'error': result['error']
        })
        
        # Show progress
        if result['score'] is not None:
            print(f"  Score: {result['score']}/5 - {result['decision']}")
        else:
            print(f"  Error: {result['error']}")
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = output_path or config.TRAINING_DATA_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved annotated dataset to {output_path}")
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Successful annotations: {df['score'].notna().sum()}")
    print(f"Failed annotations: {df['score'].isna().sum()}")
    if df['score'].notna().any():
        print(f"\nScore distribution:")
        print(df['score'].value_counts().sort_index())
        print(f"\nDecision distribution:")
        print(df['decision'].value_counts())
        print(f"\nMean score: {df['score'].mean():.2f}")
    
    return df


def main():
    """Main function to create training dataset."""
    print("=" * 60)
    print("FineWeb-Edu Training Data Generation")
    print("=" * 60)
    
    # Configuration
    NUM_SAMPLES = 100  # Number of texts to sample and annotate
    
    print(f"\nConfiguration:")
    print(f"  - Number of samples: {NUM_SAMPLES}")
    print(f"  - Teacher model: {config.TEACHER_MODEL}")
    print(f"  - Output path: {config.TRAINING_DATA_PATH}")
    
    # Step 1: Sample texts from FineWeb-Edu
    texts = sample_diverse_texts(num_samples=NUM_SAMPLES)
    
    if not texts:
        print("Error: No texts were sampled. Exiting.")
        return
    
    # Step 2: Annotate with Teacher model
    df = annotate_with_teacher(texts)
    
    print("\n" + "=" * 60)
    print("Training data generation complete!")
    print("=" * 60)
    print(f"\nNext step: Run 'python scripts/train_student.py' to train the Student classifier")


if __name__ == "__main__":
    main()

