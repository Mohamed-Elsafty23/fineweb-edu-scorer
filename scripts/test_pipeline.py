"""
Integration test for the complete FineWeb-Edu pipeline.
Tests extraction, Teacher annotation, and Student prediction on sample URLs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extractors.web_extractor import WebExtractor
from src.annotators.teacher_annotator import TeacherAnnotator
from src.classifiers.student_classifier import StudentClassifier
import config


def test_pipeline():
    """Test the complete pipeline with sample URLs."""
    print("=" * 70)
    print("FineWeb-Edu Pipeline Integration Test")
    print("=" * 70)
    
    # Test URLs with expected different educational values
    test_cases = [
        {
            'url': 'https://en.wikipedia.org/wiki/Machine_learning',
            'expected': 'high',
            'description': 'Wikipedia article (high educational value)'
        },
        {
            'url': 'https://www.python.org/about/',
            'expected': 'medium',
            'description': 'Technical documentation (medium educational value)'
        }
    ]
    
    # Initialize components
    print("\n[1/4] Initializing components...")
    extractor = WebExtractor()
    teacher = TeacherAnnotator()
    
    # Try to load Student model
    try:
        student = StudentClassifier.load()
        student_available = True
        print("  Web Extractor initialized")
        print("  Teacher Annotator initialized")
        print("  Student Classifier loaded")
    except FileNotFoundError:
        student = None
        student_available = False
        print("  Web Extractor initialized")
        print("  Teacher Annotator initialized")
        print("  Student Classifier not found (train it first)")
    
    # Run tests
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'=' * 70}")
        print(f"Test Case {i+1}/{len(test_cases)}: {test_case['description']}")
        print(f"URL: {test_case['url']}")
        print(f"Expected: {test_case['expected']} educational value")
        print('=' * 70)
        
        # Step 1: Extract text
        print("\n[2/4] Extracting text...")
        extraction_result = extractor.extract_from_url(test_case['url'])
        
        if extraction_result['error']:
            print(f"  Extraction failed: {extraction_result['error']}")
            results.append({
                'url': test_case['url'],
                'status': 'extraction_failed',
                'error': extraction_result['error']
            })
            continue
        
        text = extraction_result['text']
        print(f"  Extracted {len(text)} characters")
        print(f"  Preview: {text[:150]}...")
        
        # Step 2: Teacher annotation
        print("\n[3/4] Getting Teacher annotation...")
        print("  (This may take 10-30 seconds...)")
        teacher_result = teacher.get_educational_score(text)
        
        if teacher_result['error']:
            print(f"  Teacher annotation failed: {teacher_result['error']}")
            results.append({
                'url': test_case['url'],
                'status': 'teacher_failed',
                'error': teacher_result['error']
            })
            continue
        
        teacher_score = teacher_result['score']
        teacher_decision = teacher_result['decision']
        print(f"  Teacher Score: {teacher_score}/5")
        print(f"  Decision: {teacher_decision}")
        print(f"  Reasoning: {teacher_result['reasoning'][:150]}...")
        
        # Step 3: Student prediction
        student_score = None
        student_decision = None
        
        if student_available:
            print("\n[4/4] Getting Student prediction...")
            try:
                student_score = student.predict(text)
                student_decision = 'KEEP' if student_score >= config.EDUCATIONAL_THRESHOLD else 'DISCARD'
                print(f"  Student Score: {student_score:.2f}/5")
                print(f"  Decision: {student_decision}")
                
                # Compare
                error = abs(teacher_score - student_score)
                print(f"\n  Comparison:")
                print(f"     Teacher: {teacher_score}/5 → {teacher_decision}")
                print(f"     Student: {student_score:.2f}/5 → {student_decision}")
                print(f"     Error: {error:.2f}")
                
                if teacher_decision == student_decision:
                    print(f"     Agreement on decision!")
                else:
                    print(f"     Disagreement on decision")
                
            except Exception as e:
                print(f"  Student prediction failed: {str(e)}")
                student_score = None
        else:
            print("\n[4/4] Student prediction skipped (model not trained)")
        
        # Store results
        results.append({
            'url': test_case['url'],
            'expected': test_case['expected'],
            'status': 'success',
            'teacher_score': teacher_score,
            'teacher_decision': teacher_decision,
            'student_score': student_score,
            'student_decision': student_decision,
            'agreement': teacher_decision == student_decision if student_score else None
        })
    
    # Summary
    print(f"\n{'=' * 70}")
    print("Test Summary")
    print('=' * 70)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nTests completed: {len(results)}")
    print(f"Successful: {successful}/{len(results)}")
    
    if successful > 0:
        print("\nResults:")
        for r in results:
            if r['status'] == 'success':
                print(f"\n  {r['url'][:50]}...")
                print(f"    Expected: {r['expected']}")
                print(f"    Teacher: {r['teacher_score']}/5 → {r['teacher_decision']}")
                if r['student_score']:
                    print(f"    Student: {r['student_score']:.2f}/5 → {r['student_decision']}")
                    print(f"    Agreement: {'Yes' if r['agreement'] else 'No'}")
    
    print(f"\n{'=' * 70}")
    if successful == len(results):
        print("All tests passed!")
    elif successful > 0:
        print("Some tests passed, check errors above")
    else:
        print("All tests failed")
    print('=' * 70)


if __name__ == "__main__":
    test_pipeline()

