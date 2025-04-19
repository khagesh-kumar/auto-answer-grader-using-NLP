import streamlit as st
import pandas as pd
import re
import string
import nltk
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import os

# Download necessary NLTK resources
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Check and load Spacy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    st.error("Spacy model 'en_core_web_md' not found. Please install it by running:")
    st.code("python -m spacy download en_core_web_md")
    st.stop()

# Function to clean text
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to correct spelling errors
def correct_typos(text):
    if not text:
        return ""
    try:
        return str(TextBlob(text).correct())
    except:
        return text  # Return original if correction fails

# Function to get the WordNet POS tag for lemmatization
def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Function to lemmatize text
def lemmatize_text(text):
    if not text:
        return ""
    lemmatizer = WordNetLemmatizer()
    pos_tagged = pos_tag(word_tokenize(text))
    wordnet_tagged = [(word, pos_tagger(tag)) for word, tag in pos_tagged]
    return " ".join([lemmatizer.lemmatize(word, tag) if tag else word for word, tag in wordnet_tagged])

# Function to calculate text similarity
def calculate_similarity(original, student):
    if not original or not student:
        return 0.0
    
    cv = CountVectorizer(stop_words='english')
    try:
        data_dtm = cv.fit_transform([original, student])
        data_df = pd.DataFrame(data_dtm.toarray(), columns=cv.get_feature_names_out())
        common_word_count = sum((data_df.iloc[0] != 0) & (data_df.iloc[1] != 0))
        total_words_student = sum(data_df.iloc[1] != 0)
        return common_word_count / total_words_student if total_words_student else 0.0
    except ValueError:
        return 0.0

# Function to predict score based on similarity
def predict_score(original, student):
    if not original or not student:
        return 0.0

    try:
        original_clean = clean_text(original)
        student_clean = clean_text(student)
        
        original_fixed = correct_typos(original_clean)
        student_fixed = correct_typos(student_clean)
        
        original_lem = lemmatize_text(original_fixed)
        student_lem = lemmatize_text(student_fixed)
        
        similarity = calculate_similarity(original_lem, student_lem)
        predicted_score = max(0.0, min(100.0, 90.0 * similarity + 10.0))  # Added base score of 10
        
        return predicted_score
    except Exception as e:
        st.error(f"Error during score prediction: {str(e)}")
        return 0.0

# Streamlit App
def main():
    st.title("Automated Evaluation of Descriptive Answer Scripts via NLP")
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown("""
    1. Upload two text files:
       - First file: Original/Model answer
       - Second file: Student's answer
    2. Click the Evaluate button
    3. View the predicted score and grade
    """)
    
    # Sample data section
    if st.sidebar.checkbox("Use sample data"):
        sample_dir = "sample_inputs"
        if os.path.exists(sample_dir) and os.path.isdir(sample_dir):
            sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.txt')]
            if len(sample_files) >= 2:
                with open(os.path.join(sample_dir, "original.txt"), 'r') as f:
                    sample_original = f.read()
                with open(os.path.join(sample_dir, "student.txt"), 'r') as f:
                    sample_student = f.read()
                
                st.text_area("Original/Model Answer (Sample)", sample_original, height=200)
                st.text_area("Student Answer (Sample)", sample_student, height=200)
                
                if st.button("Evaluate Sample"):
                    score = predict_score(sample_original, sample_student)
                    display_results(score)
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload two text files (original and student response)",
        accept_multiple_files=True, 
        type=['txt'],
        key="file_uploader"
    )
    
    if len(uploaded_files) == 2 and st.button("Evaluate"):
        try:
            original_text = uploaded_files[0].read().decode("utf-8").strip()
            student_text = uploaded_files[1].read().decode("utf-8").strip()
            
            if not original_text or not student_text:
                st.error("Both files must contain text!")
                return
            
            score = predict_score(original_text, student_text)
            display_results(score)
            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")

def display_results(score):
    st.subheader("Evaluation Results")
    st.write("**Predicted Score:**", round(score, 2))
    
    grade = "O" if score >= 90 else \
            "A+" if score >= 80 else \
            "A" if score >= 70 else \
            "B+" if score >= 60 else \
            "B" if score >= 50 else \
            "C" if score >= 40 else "F"
    
    st.write("**Grade:**", grade)
    
    # Visualization
    st.progress(score/100)
    st.caption(f"Score Visualization: {score:.1f}/100")

# Run the app
if __name__ == '__main__':
    main()