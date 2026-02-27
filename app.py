import streamlit as st
import pickle as pkl
from nltk.corpus import stopwords
import nltk
import re

# Load the model
model = pkl.load(open('model.pkl', 'rb'))

# Load required objects
stemmer = pkl.load(open('stemmer.pkl', 'rb'))
vectorizer = pkl.load(open('vectorizer.pkl', 'rb'))

# Download stopwords 
nltk.download('stopwords')

# Function to preprocess text
def stemming(data):
  stemmed_content = re.sub('[^a-zA-Z]', ' ', data)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
  stemmed_content = ' '.join(stemmed_content)

  return stemmed_content

# Create streamlit app
st.title('Twitter Sentiment Analysis')
# Get user input
user_input = st.text_input('Enter a tweet to analyze its sentiment:')
if st.button('Analyze'):
    if user_input:
        # Preprocess the input
        processed_input = stemming(user_input)
        # Vectorize the input and make prediction
        vectorized_input = vectorizer.transform([processed_input])
        # Predict sentiment
        prediction = model.predict(vectorized_input)
        # Display result
        if prediction[0] == 0:
          st.error('The sentiment of the tweet is: Negative')
        else:
          st.success('The sentiment of the tweet is: Positive')
    else:
        st.warning('Please enter a tweet to analyze.')