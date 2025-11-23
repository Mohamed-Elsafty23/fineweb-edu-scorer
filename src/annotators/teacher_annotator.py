import re
from typing import Dict, Optional
from src.utils.api_client import api_client
import config


class TeacherAnnotator:
    
    EDUCATIONAL_PROMPT_TEMPLATE = """Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below.

Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some information related to any topic, even if it includes unrelated or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula, even if it does not follow a traditional textbook structure. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:
{text}

After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score: <total points>"
"""
    
    def __init__(self, api_client_instance=None):
        self.api_client = api_client_instance or api_client
    
    def truncate_text(self, text: str, max_words: int = 2000) -> str:
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + "..."
        return text
    
    def parse_score(self, response: str) -> Optional[int]:
        patterns = [
            r"Educational score:\s*(\d+)",
            r"educational score:\s*(\d+)",
            r"Total score:\s*(\d+)",
            r"Score:\s*(\d+)",
            r"(\d+)\s*points?",
            r"(\d+)/5"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 0 <= score <= 5:
                    return score
        
        return None
    
    def get_educational_score(self, text: str) -> Dict:
        result = {
            'score': None,
            'reasoning': None,
            'full_response': None,
            'decision': None,
            'error': None
        }
        
        try:
            truncated_text = self.truncate_text(text)
            
            prompt = self.EDUCATIONAL_PROMPT_TEMPLATE.format(text=truncated_text)
            
            response = self.api_client.get_llm_response(
                prompt=prompt,
                system_message="You are an expert educational content evaluator. Analyze the text carefully and provide a fair, accurate assessment.",
                temperature=0.3,
                max_tokens=500
            )
            
            result['full_response'] = response
            
            score = self.parse_score(response)
            
            if score is None:
                result['error'] = "Could not parse score from LLM response"
                result['reasoning'] = response
            else:
                result['score'] = score
                result['reasoning'] = response
                result['decision'] = 'KEEP' if score >= config.EDUCATIONAL_THRESHOLD else 'DISCARD'
            
        except Exception as e:
            result['error'] = f"Error getting educational score: {str(e)}"
        
        return result
    
    def annotate_batch(self, texts: list) -> list:
        annotations = []
        for i, text in enumerate(texts):
            print(f"Annotating text {i+1}/{len(texts)}...")
            annotation = self.get_educational_score(text)
            annotations.append(annotation)
        return annotations


def test_teacher_annotator():
    print("Testing Teacher Annotator...")
    
    annotator = TeacherAnnotator()
    
    test_texts = [
        """Machine learning is a subset of artificial intelligence that enables computers to learn from data 
        without being explicitly programmed. It involves algorithms that can identify patterns in data and make 
        predictions or decisions based on those patterns. The three main types of machine learning are supervised 
        learning, unsupervised learning, and reinforcement learning.""",
        
        """Click here now! Best deals on products you don't need! Buy now and save! 
        Limited time offer! Subscribe to our newsletter!"""
    ]
    
    for i, text in enumerate(test_texts):
        print(f"\n\nTest {i+1}:")
        print(f"Text preview: {text[:100]}...")
        result = annotator.get_educational_score(text)
        
        if result['error']:
            print(f"Error: {result['error']}")
        else:
            print(f"Score: {result['score']}/5")
            print(f"Decision: {result['decision']}")
            print(f"Reasoning: {result['reasoning'][:200]}...")


if __name__ == "__main__":
    test_teacher_annotator()
