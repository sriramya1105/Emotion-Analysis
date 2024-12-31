import streamlit as st
import pandas as pd
from langdetect import detect
from deep_translator import GoogleTranslator
import re
import nltk
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
# Load BERT-based emotion detection model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

# Updated Emotion-to-Image Mapping (All emotions included)
emotion_images = {
    "admiration": "assets/admiration.png",
    "amusement": "assets/amusement.png",
    "anger": "assets/anger.png",
    "annoyance": "assets/annoyance.png",
    "approval": "assets/approval.png",
    "boredom": "assets/boredom.jpeg",
    "caring": "assets/caring.avif",
    "confusion": "assets/confusion.png",
    "curiosity": "assets/curiosity.jpeg",
    "desire": "assets/desire.jpeg",
    "disapproval": "assets/disapproval.png",
    "disgust": "assets/disgust.png",
    "disappointment": "assets/dissapointment.jpeg",
    "embarrassment": "assets/embarrassment.png",
    "envy": "assets/envy.jpeg",
    "excitement": "assets/excitement.png",
    "fear": "assets/fear.jpg",
    "gratitude": "assets/gratitude.png",
    "grief": "assets/grief.jpeg",
    "hope": "assets/hope.jpeg",
    "joy": "assets/joy.jpg",
    "love": "assets/love.jpg",
    "nervousness": "assets/nervousness.jpeg",
    "neutral": "assets/Neutral.jpeg",
    "optimism": "assets/optimism.png",
    "pride": "assets/pride.jpeg",
    "realisation": "assets/realisation.jpeg",
    "relief": "assets/relief.jpeg",
    "remorse": "assets/remorse.jpeg",
    "sadness": "assets/sad.jpg",
    "surprise": "assets/surprise.jpg"
}

# Synonyms mapping for emotions
emotion_synonyms = {
    "sadness": ["sorrow", "unhappy", "grief", "heartbroken", "mournful"],
    "joy": ["happiness", "cheerful", "delight", "pleasure", "contentment"],
    "love": ["affection", "fondness", "adoration", "devotion", "attachment"],
    "anger": ["rage", "fury", "wrath", "irritation", "resentment"],
    "fear": ["anxiety", "worry", "nervousness", "panic", "terror"],
    "surprise": ["astonishment", "amazement", "shock", "bewilderment", "startle"],
    "neutral": ["indifferent", "apathetic", "uninterested", "neutral", "bland"],
    "admiration": ["respect", "esteem", "appreciation", "regard", "approval"],
    "amusement": ["entertainment", "fun", "enjoyment", "humor", "delight"],
    "annoyance": ["irritation", "bother", "displeasure", "vexation", "exasperation"],
    "approval": ["acceptance", "endorsement", "appreciation", "agreement", "praise"],
    "boredom": ["disinterest", "monotony", "dullness", "tedium", "ennui"],
    "caring": ["compassion", "concern", "kindness", "sympathy", "love"],
    "confusion": ["perplexity", "bewilderment", "puzzlement", "disorientation", "uncertainty"],
    "curiosity": ["interest", "inquisitiveness", "wonder", "intrigue", "eagerness"],
    "desire": ["longing", "wish", "yearning", "craving", "want","would"],
    "disapproval": ["dislike", "condemnation", "rejection", "objection", "dissatisfaction"],
    "disgust": ["revulsion", "dislike", "aversion", "repulsion", "abhorrence"],
    "embarrassment": ["self-consciousness", "shame", "awkwardness", "humiliation", "discomfort"],
    "envy": ["jealousy", "covetousness", "resentment", "greed", "longing"],
    "excitement": ["enthusiasm", "thrill", "eagerness", "joy", "elation"],
    "grief": ["sorrow", "mourning", "sadness", "lamentation", "heartache"],
    "hope": ["expectation", "optimism", "faith", "trust", "anticipation"],
    "optimism": ["hopefulness", "positivity", "confidence", "faith", "expectancy"],
    "pride": ["self-esteem", "dignity", "honor", "satisfaction", "achievement"],
    "realisation": ["awareness", "recognition", "realization", "understanding", "comprehension"],
    "relief": ["ease", "comfort", "release", "freedom", "calm"],
    "remorse": ["guilt", "sorrow", "regret", "contrition", "penitence"],
    "nervousness": ["anxiety", "apprehension", "unease", "restlessness", "fear"],
}

# Detect language and translate if necessary
def detect_and_translate(text):
    detected_language = detect(text)
    if detected_language != 'en':
        translator = GoogleTranslator(source=detected_language, target='en')
        translated_text = translator.translate(text)
    else:
        translated_text = text
    return translated_text

# Negation Handling for Emotion Adjustment
def handle_negations(text, original_emotion):
    negation_words = ["not", "no", "never", "don't", "isn't", "aren't", "won't", "can't", "didn't", "doesn't"]
    text_tokens = nltk.word_tokenize(text.lower())
    contains_negation = any(word in text_tokens for word in negation_words)

    # Define negation-based emotion adjustments
    emotion_flip_map = {
        "sadness": "joy",           # "I am not sad" -> joy
        "joy": "sadness",           # "I am not happy" -> sadness
        "love": "disappointment",   # "I do not love this" -> disappointment
        "admiration": "disapproval",# "I am not impressed" -> disapproval
        "approval": "disapproval",  # "I do not approve" -> disapproval
        "anger": "calm",            # "I am not angry" -> calm
        "fear": "confidence",       # "I am not scared" -> confidence
        "disgust": "admiration",    # "I am not disgusted" -> admiration
        "annoyance": "amusement",   # "I am not annoyed" -> amusement
        "optimism": "disappointment",# "I am not optimistic" -> disappointment
        "excitement": "fear",       # "I am not excited" -> fear
    }

    # Adjust emotion if negation is detected
    if contains_negation and original_emotion in emotion_flip_map:
        return emotion_flip_map[original_emotion]
    return original_emotion

# Analyze sentiment and emotion considering synonyms
def analyze_emotion_and_sentiment(text):
    # Classify emotion
    result = emotion_classifier(text)
    original_emotion = result[0]['label'] if result else "neutral"
    confidence = result[0]['score'] if result else 0

    # Check for synonyms
    for emotion, synonyms in emotion_synonyms.items():
        if any(synonym in text.lower() for synonym in synonyms):
            original_emotion = emotion
            break

    # Handle negation and adjust emotion
    adjusted_emotion = handle_negations(text, original_emotion)

    # Define sentiment based on emotion
    positive_emotions = ["admiration", "amusement", "approval", "joy", "love", "optimism", "pride", "relief", "gratitude", "excitement", "hope","desire"]
    sentiment = "positive" if adjusted_emotion in positive_emotions else "negative"

    return adjusted_emotion, sentiment, confidence

# Validate review input
def is_valid_review(review, min_length=10):
    # Check if the review is too short (below the minimum length)
    if len(review.strip()) < min_length:
        return False
    
    # Invalid review check for combinations of numbers, characters, and special symbols
    if re.match(r'^[0-9]+$', review) or re.match(r'^[!@#$%^&*(),.?":{}|<>]+$', review):
        return False
    
    # Check for common short phrases or greetings (e.g., 'hi', 'hello', 'good morning')
    invalid_phrases = ["hi", "hello", "good morning", "hey", "howdy", "greetings", "good evening"]
    if review.strip().lower() in invalid_phrases:
        return False
    
    return True

# Plot bar chart for emotions
def plot_emotion_bar_chart(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    emotion_counts = data.value_counts()
    emotion_counts.plot(kind='bar', ax=ax, color=sns.color_palette("Set2", len(emotion_counts.unique())))
    ax.set_ylabel('Count')
    ax.set_xlabel('Emotion')
    ax.set_title('Emotion Distribution')
    st.pyplot(fig)

# Plot pie chart for sentiments
def plot_sentiment_pie_chart(data):
    sentiment_counts = data.value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(sentiment_counts.unique())))
    ax.set_ylabel('')
    ax.set_title('Sentiment Distribution')
    st.pyplot(fig)

# Streamlit App
st.image("assets/AVN logo.jpg", width=800) 
st.title('Emotion-Driven Multilingual Sentiment Analysis Web Application using NLP')

st.subheader("User Guide")
st.write("""
This app performs sentiment analysis on user input text or reviews in a CSV file. 
It includes the following features:

1. **Text Sentiment Analysis**: Detects the language, translates it (if needed), and predicts emotions.
2. **Bulk Analysis**: Upload a CSV file containing reviews, and the app will predict emotions for each review.
3. **Emotion Count Summary**: View the count of each emotion predicted for the uploaded reviews.
4. **Download Results**: After analysis, you can download the results as a CSV file with predictions.

### Example:
**Input (Hindi)**: "यह फिल्म बहुत शानदार थी, मुझे इसकी कहानी और अभिनय बहुत पसंद आया।"  
**Detected Language**: Hindi  
**Translated Text**: "This movie was very amazing, I really liked its story and acting."  
**Predicted Emotion**: Love
""")


# Sidebar
st.sidebar.image("assets/team.jpeg", width=300)
st.sidebar.title("Project Details")
st.sidebar.write("### AIML Department")
st.sidebar.write("**Department Head: Dr.M.Jayaram (Professor)**")
st.sidebar.write("**Project Guide: CH.Jyothi (Assistant Professor)**")
st.sidebar.write("**Project Batch: A-12**")
st.sidebar.write("**Team Members:**")
st.sidebar.write("- K. Sri Ramya")
st.sidebar.write("- Chikkapalli Lavanya")
st.sidebar.write("- Endapally Dinesh")
st.sidebar.write("- Kompalli Mahesh")

# Single Review Analysis
st.subheader("Analyze a Single Review")
user_input = st.text_area("Enter a review:")

if st.button("Analyze Review"):
    if user_input.strip():
        if is_valid_review(user_input):
            translated_text = detect_and_translate(user_input)
            emotion, sentiment, confidence = analyze_emotion_and_sentiment(translated_text)

            st.success(f"Emotion: {emotion.capitalize()} (Confidence: {confidence:.2f})")
            st.success(f"Sentiment: {sentiment.capitalize()}")

            # Display corresponding image for emotion
            if emotion in emotion_images:
                st.image(emotion_images[emotion], caption=emotion.capitalize())
        else:
            st.error(f"Invalid review. Please provide a valid review.")
    else:
        st.warning("Please enter a review.")

# Bulk Analysis
st.subheader("Bulk Analysis of Reviews")
uploaded_file = st.file_uploader("Upload a CSV file with a text column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Dynamically choose the review column
    review_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['review', 'feedback', 'text'])]
    
    if review_columns:
        text_column = st.selectbox("Select the Review Column", review_columns)
    else:
        st.warning("No appropriate review column found. Please check your dataset.")
        text_column = None
    
    if text_column:
        # Validate if the column contains strings (reviews)
        if not df[text_column].apply(lambda x: isinstance(x, str)).all():
            st.error(f"The column '{text_column}' does not contain valid text data.")
        else:
            reviews = df[text_column].tolist()
            emotions, sentiments = [], []

            for review in reviews:
                if is_valid_review(review):
                    translated_text = detect_and_translate(review)
                    emotion, sentiment, _ = analyze_emotion_and_sentiment(translated_text)
                    emotions.append(emotion)
                    sentiments.append(sentiment)
                else:
                    emotions.append("invalid")
                    sentiments.append("invalid")

            df['Emotion'] = emotions
            df['Sentiment'] = sentiments

            # Emotion and sentiment count summaries
            emotion_summary = df['Emotion'].value_counts().reset_index().rename(columns={'index': 'Emotion', 'Emotion': 'Count'})
            sentiment_summary = df['Sentiment'].value_counts().reset_index().rename(columns={'index': 'Sentiment', 'Sentiment': 'Count'})

            # Display Review with Emotion and Sentiment Columns
            st.write("### Review, Emotion, and Sentiment Summary")
            st.write(df[[text_column, 'Emotion', 'Sentiment']])  # Use the dynamic column name

            # Emotion and Sentiment count tables
            st.write("### Emotion Count Summary")
            st.write(emotion_summary)

            st.write("### Sentiment Count Summary")
            st.write(sentiment_summary)

            # Visualizations
            st.write("### Emotion Count Distribution")
            plot_emotion_bar_chart(df['Emotion'])

            st.write("### Sentiment Count Distribution")
            plot_sentiment_pie_chart(df['Sentiment'])

            # Download button for CSV results
            st.download_button("Download Results as CSV", df.to_csv(index=False), "analysis_results.csv")

