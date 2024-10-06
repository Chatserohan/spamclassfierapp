import streamlit as st 
import pickle 
import pandas as pd
import re
import string

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer1.pkl','rb'))
model = pickle.load(open('model1.pkl','rb'))

# Streamlit title and input
st.title('Email/SMS Spam Classifier')
input_sms = st.text_area('Enter the message')


if st.button('Predict'):


    # Sample stopwords list
    stop_words = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
        "at", "by", "for", "with", "about", "against", "between", "into", "through", 
        "during", "before", "after", "above", "below", "to", "from", "up", "down", 
        "in", "out", "on", "off", "over", "under", "again", "further", "then", 
        "once", "here", "there", "when", "where", "why", "how", "all", "any", 
        "both", "each", "few", "more", "most", "other", "some", "such", "no", 
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", 
        "t", "can", "will", "just", "don", "should", "now"
    ])

    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation using regex
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # Tokenize using regex to split on whitespace
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove stop words
        processed_tokens = [token for token in tokens if token not in stop_words]
        
        # Join tokens back into a single string
        return ' '.join(processed_tokens)

    # Preprocess the input message
    if input_sms:
        transformed_sms = preprocess_text(input_sms)

        # Vectorize the input
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.header('Spam')
        else:
            st.header('Not Spam')
