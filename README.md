# 🐦 Twitter Sentiment Analysis Web App

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154734?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

A Machine Learning–based Twitter Sentiment Analysis Web Application that classifies tweets as Positive or Negative using Natural Language Processing (NLP).

Built using Python, Scikit-learn, NLTK, and Streamlit, and deployed online for real-time sentiment prediction.

---

## 🔗 Live Demo

👉 Try the deployed web app here: https://twitter-sentiment-analysis-by-saikat-pradhan.streamlit.app/

---

## 🚀 Project Overview

- ### This project demonstrates how Machine Learning and NLP techniques can analyze textual data and classify sentiment.

- ### Users can:

- Enter any tweet text

- Click Analyze

- Instantly get sentiment prediction (Positive / Negative)

- ### The model preprocesses text, removes stopwords, applies stemming, vectorizes the text, and predicts sentiment using a trained ML model.

---

## 🎯 Application Features

- Real-time tweet sentiment analysis
- Text preprocessing (cleaning & normalization)
- Stopword removal
- Stemming using NLP
- TF-IDF Vectorization
- Machine Learning classification
- Clean and interactive Streamlit UI
- Deployed on Streamlit Cloud

---

## 🧠 Technologies Used

- Python 🐍
- Streamlit 🌐
- Scikit-learn 🤖
- NLTK 📚
- Regex 🔍
- Pickle 📦
- Pandas 📊
- NumPy 📐

---

## 📊 Dataset

### Dataset used for training:

🔗 Kaggle Dataset: Sentiment140 dataset with 1.6 million tweets
https://www.kaggle.com/datasets/kazanova/sentiment140

### Dataset Details:

- 1.6 Million Tweets
- Labeled Sentiment:
 - 0 → Negative
 - 4 → Positive
- Columns include:
- Target (sentiment)
- Tweet ID
- Date
- Query
- User
- Tweet Text

This large dataset helps the model learn real-world sentiment patterns effectively.

---

## 🏗️ Model Training

Model development is performed in:

📓 Twitter_Sentiment_Analysis_Using_ML.ipynb

### Training Steps:

- Data Loading
- Data Cleaning
- Removing Special Characters
- Converting to Lowercase
- Stopword Removal
- Stemming
- TF-IDF Vectorization
- Train-Test Split
- Model Training (Logistic Regression)
- Model Evaluation
- Saving Model using Pickle

---

## 💾 Saved Files

- model.pkl → Trained ML Model
- vectorizer.pkl → TF-IDF Vectorizer
- stemmer.pkl → NLP Stemmer

---

## 🧠 How the App Works

- User enters a tweet.
- Text is cleaned using Regex.
- Stopwords are removed.
- Words are stemmed.
- Text is transformed using TF-IDF Vectorizer.
- Trained ML model predicts sentiment.
- Result is displayed as:

  - ❌ Negative
  - ✅ Positive

---

## 📁 Project Structure
```
Twitter-Sentiment-Analysis
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── stemmer.pkl
├── requirements.txt
├── Twitter_Sentiment_Analysis_Using_ML.ipynb
└── README.md
```

---

## ⚙️ Setup Guide (Run Locally)
### 1️⃣ Clone the Repository
```
git clone https://github.com/Saikat-Pradhan/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
```
### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit App
```
streamlit run app.py
```
### Then open your browser:
```
http://localhost:8501
```

---

## 🌍 Deployment

 Successfully deployed using Streamlit Cloud

### Live URL:
https://twitter-sentiment-analysis-by-saikat-pradhan.streamlit.app/

--- 

## 📈 Example Usage
```
Input: "I absolutely love this product!"
Output: The sentiment of the tweet is: Positive
```
```
Input: "This is the worst experience ever."
Output: The sentiment of the tweet is: Negative
```

---

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub.

It motivates me to build more Machine Learning & AI projects 🚀

---

## 👨‍💻 Author

Saikat Pradhan

🔗 GitHub: https://github.com/Saikat-Pradhan
