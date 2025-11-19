"""
FineWeb-Edu: Educational Content Scorer - Interactive Demo
A three-phase pipeline for scoring educational web content using Knowledge Distillation
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.extractors.web_extractor import WebExtractor
from src.annotators.teacher_annotator import TeacherAnnotator
from src.classifiers.student_classifier import StudentClassifier
import config

st.set_page_config(
    page_title="FineWeb-Edu: Educational Content Scorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def initialize_models():
    """Initialize models and cache them in session state."""
    if 'extractor' not in st.session_state:
        st.session_state.extractor = WebExtractor(max_text_length=1000000)  # Remove limit
    
    if 'teacher' not in st.session_state:
        st.session_state.teacher = TeacherAnnotator()
    
    if 'student' not in st.session_state:
        try:
            st.session_state.student = StudentClassifier.load()
            st.session_state.student_loaded = True
        except FileNotFoundError:
            st.session_state.student = None
            st.session_state.student_loaded = False


def main():
    """Main application function."""
    
    initialize_models()
    
    # Professional header
    st.title("📚 FineWeb-Edu: Educational Content Scorer")
    st.markdown("**Interactive Demo** | Three-Phase Knowledge Distillation Pipeline")
    st.markdown("*Replicating the methodology from \"The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale\"*")
    
    st.divider()
    
    # Overview of the three phases
    st.markdown("### Pipeline Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Phase 1: Extraction**\nClean text from web pages")
    with col2:
        st.info("**Phase 2: Teacher Scoring**\nLLM evaluates educational value")
    with col3:
        st.info("**Phase 3: Student Prediction**\nFast classifier mimics Teacher")
    
    st.divider()
    
    # Input options
    st.markdown("### Input Source")
    input_method = st.radio(
        "Choose your input method:",
        ["URL", "Direct Text"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if input_method == "URL":
        # Initialize session state for URL
        if 'url_input_field' not in st.session_state:
            st.session_state.url_input_field = ""
        
        # Example URLs
        st.markdown("**Quick Examples:**")
        col1, col2, col3 = st.columns(3)
        
        example_urls = {
            "Wikipedia (High)": "https://en.wikipedia.org/wiki/Machine_learning",
            "Python Docs (Medium)": "https://docs.python.org/3/tutorial/index.html",
            "News Article (Low)": "https://www.bbc.com/news"
        }
        
        with col1:
            if st.button("📚 Wikipedia Article", use_container_width=True, help="High educational value", key="btn_url_1"):
                st.session_state.url_input_field = example_urls["Wikipedia (High)"]
        with col2:
            if st.button("📖 Python Tutorial", use_container_width=True, help="Medium educational value", key="btn_url_2"):
                st.session_state.url_input_field = example_urls["Python Docs (Medium)"]
        with col3:
            if st.button("📰 News Website", use_container_width=True, help="Low educational value", key="btn_url_3"):
                st.session_state.url_input_field = example_urls["News Article (Low)"]
        
        # URL input - Streamlit automatically syncs with session state via key
        url_input = st.text_input(
            "Enter URL:",
            placeholder="https://en.wikipedia.org/wiki/Machine_learning",
            help="Paste any webpage URL to analyze its educational content",
            key="url_input_field"
        )
        
        input_provided = bool(url_input)
        input_value = url_input
    else:
        # Initialize session state for text
        if 'text_input_field' not in st.session_state:
            st.session_state.text_input_field = ""
        
        # Example texts
        st.markdown("**Quick Examples:**")
        col1, col2, col3 = st.columns(3)
        
        example_texts = {
            "Educational": """Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns. The three main types of machine learning are supervised learning, where models learn from labeled data; unsupervised learning, where models find patterns in unlabeled data; and reinforcement learning, where agents learn through interaction with an environment.""",
            
            "Mixed Quality": """Check out our amazing products! We have the best deals on electronics. Machine learning is used in many applications today. Buy now and save 50%! Free shipping on all orders over $50. Our AI-powered recommendation system helps you find what you need. Click here to shop now!""",
            
            "Non-Educational": """CLICK HERE NOW! Limited time offer! Best deals of the year! Don't miss out! Buy now and save big! Exclusive discounts just for you! Shop today! Amazing prices! Free shipping! Order now! Hurry, while supplies last! Special promotion! Act fast!"""
        }
        
        with col1:
            if st.button("📚 Educational Text", use_container_width=True, help="High quality educational content", key="btn_text_1"):
                st.session_state.text_input_field = example_texts["Educational"]
        with col2:
            if st.button("📊 Mixed Content", use_container_width=True, help="Educational mixed with promotional", key="btn_text_2"):
                st.session_state.text_input_field = example_texts["Mixed Quality"]
        with col3:
            if st.button("🛍️ Promotional Text", use_container_width=True, help="Non-educational spam", key="btn_text_3"):
                st.session_state.text_input_field = example_texts["Non-Educational"]
        
        # Text input - Streamlit automatically syncs with session state via key
        text_input = st.text_area(
            "Paste your text here:",
            height=200,
            placeholder="Paste any text content you want to evaluate for educational value...",
            help="Paste text content directly for analysis",
            key="text_input_field"
        )
        
        input_provided = bool(text_input)
        input_value = text_input
    
    if input_provided:
        analyze_button = st.button("🚀 Analyze Content", type="primary", use_container_width=True)
        
        if analyze_button:
            st.divider()
            
            # ============ PHASE 1: TEXT EXTRACTION ============
            st.markdown("## Phase 1: Text Extraction")
            st.caption("*Internal process: Using Trafilatura to remove boilerplate (ads, navigation, etc.)*")
            
            if input_method == "URL":
                with st.spinner("📡 Fetching webpage and extracting content..."):
                    result = st.session_state.extractor.extract_from_url(input_value)
                    
                if result['error']:
                    st.error(f"❌ Extraction failed: {result['error']}")
                    return
                
                # Show comparison
                st.success(f"✅ Successfully extracted {len(result['text'])} characters")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Raw HTML (first 1000 chars)**")
                    st.code(result['raw_html'][:1000] if result['raw_html'] else "No preview", language='html', line_numbers=False)
                
                with col2:
                    st.markdown("**Cleaned Text (first 1000 chars)**")
                    st.text_area("", result['text'][:1000], height=200, label_visibility="collapsed", disabled=True)
                
                with st.expander("📄 View full cleaned text"):
                    st.text_area("Full content", result['text'], height=400, label_visibility="collapsed", disabled=True)
                
                st.session_state.current_text = result['text']
                st.session_state.current_url = input_value
            else:
                # Direct text input
                st.info(f"✅ Using provided text ({len(input_value)} characters)")
                st.session_state.current_text = input_value
                st.session_state.current_url = "Direct input"
            
            st.divider()
            
            # ============ PHASE 2: TEACHER SCORING ============
            st.markdown("## Phase 2: Teacher Model (LLM) Scoring")
            st.caption("*Internal process: Llama-3.3-70B evaluates content using educational rubric (Appendix F.1)*")
            
            with st.spinner("🤖 Teacher LLM analyzing content... (10-30 seconds)"):
                annotation = st.session_state.teacher.get_educational_score(st.session_state.current_text)
                st.session_state.teacher_annotation = annotation
            
            if annotation['error']:
                st.error(f"❌ Scoring failed: {annotation['error']}")
                return
            
            score = annotation['score']
            
            # Display results professionally
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Score")
                st.markdown(f"<h1 style='text-align: center; color: {'green' if score >= 3 else 'red'};'>{score}/5</h1>", unsafe_allow_html=True)
                st.progress(score / 5)
                
                if score >= config.EDUCATIONAL_THRESHOLD:
                    st.success(f"✅ **KEEP** (≥{config.EDUCATIONAL_THRESHOLD})")
                else:
                    st.error(f"❌ **DISCARD** (<{config.EDUCATIONAL_THRESHOLD})")
            
            with col2:
                st.markdown("### Educational Value Assessment")
                st.write(annotation['reasoning'])
            
            with st.expander("🔍 View detailed LLM response"):
                st.code(annotation['full_response'])
            
            st.divider()
            
            # ============ PHASE 3: STUDENT PREDICTION ============
            st.markdown("## Phase 3: Student Model (Fast Classifier)")
            st.caption("*Internal process: Ridge regression on E5-Large embeddings (1024-dim) predicts score*")
            
            if st.session_state.student_loaded:
                with st.spinner("⚡ Student model predicting... (2-5 seconds)"):
                    prediction = st.session_state.student.predict(st.session_state.current_text)
                    st.session_state.student_prediction = prediction
                
                # Comparison
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Teacher Score", f"{score}/5", help="Slow but accurate LLM")
                
                with col2:
                    st.metric("Student Score", f"{prediction:.2f}/5", help="Fast classifier")
                
                with col3:
                    error = abs(prediction - score)
                    st.metric("Difference", f"{error:.2f}", 
                             delta=f"{'-' if error < 0.5 else '+'}{error:.2f}",
                             delta_color="inverse")
                
                # Agreement analysis
                teacher_decision = "KEEP" if score >= config.EDUCATIONAL_THRESHOLD else "DISCARD"
                student_decision = "KEEP" if prediction >= config.EDUCATIONAL_THRESHOLD else "DISCARD"
                
                if teacher_decision == student_decision:
                    st.success(f"✅ **Models Agree**: Both predict {teacher_decision}")
                else:
                    st.warning(f"⚠️ **Models Disagree**: Teacher: {teacher_decision} | Student: {student_decision}")
                
                # Performance insight
                st.markdown("### Knowledge Distillation Performance")
                if error <= 0.5:
                    st.info("🌟 **Excellent**: Student closely replicates Teacher's scoring (MAE ≤ 0.5)")
                elif error <= 1.0:
                    st.info("✓ **Good**: Student prediction is reasonable (MAE ≤ 1.0)")
                else:
                    st.warning("⚠️ **Fair**: Significant difference - Student may need more training data")
            else:
                st.warning("⚠️ **Student model not trained**")
                st.info("""
                To train the Student classifier:
                ```bash
                python scripts/create_training_data.py  # Generate training data
                python scripts/train_student.py         # Train the model
                ```
                """)
            
            st.divider()
            
            # Summary
            st.markdown("## Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.markdown("**Input Source:**")
                st.write(f"Type: {input_method}")
                if input_method == "URL":
                    st.write(f"URL: {st.session_state.current_url}")
                st.write(f"Length: {len(st.session_state.current_text)} characters")
            
            with summary_col2:
                st.markdown("**Final Decision:**")
                if score >= config.EDUCATIONAL_THRESHOLD:
                    st.success(f"✅ **Educational Content** (Score: {score}/5)")
                else:
                    st.error(f"❌ **Non-Educational Content** (Score: {score}/5)")
                
                if st.session_state.student_loaded:
                    st.write(f"Student model: {prediction:.2f}/5 ({student_decision})")
    
    # Footer
    st.divider()
    st.markdown("---")
    st.caption("**Implementation**: Based on 'The FineWeb Datasets' paper (Penedo et al., 2024) | **Models**: Llama-3.3-70B (Teacher), Ridge Regression on E5-Large (Student)")


if __name__ == "__main__":
    main()
