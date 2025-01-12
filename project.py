import streamlit as st
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import re
import nltk
import pandas as pd
from transformers import pipeline
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import chardet
from io import BytesIO

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load BERT-based emotion detection model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")

# Emotion-to-Image Mapping
emotion_images = {
    "admiration": "assets/admiration.png",
    "amusement": "assets/amusement.png",
    "anger": "assets/anger.png",
    "annoyance": "assets/annoyance.png",
    "approval": "assets/approval.png",
    "boredom": "assets/boredom.jpeg",
    "caring": "assets/Caring.png",
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
    "surprise": "assets/surprise.jpg",
    "hate": "assets/hate.jpeg"
}

# Synonyms mapping for emotions
emotion_synonyms = {
    "sadness": ["sorrow", "unhappy", "grief", "heartbroken", "mournful", "unsatisfied"],
    "joy": ["happiness", "cheerful", "delight", "pleasure", "contentment", "satisfied"],
    "love": ["affection", "fondness", "adoration", "devotion", "attachment", "love"],
    "anger": ["rage", "fury", "wrath", "irritation", "resentment"],
    "fear": ["anxiety", "worry", "nervousness", "panic", "terror"],
    "surprise": ["astonishment", "amazement", "shock", "bewilderment", "startle"],
    "neutral": ["indifferent", "apathetic", "uninterested", "neutral", "bland"],
    "admiration": ["respect", "esteem", "appreciation", "regard", "approval"],
    "amusement": ["entertainment", "fun", "enjoyment", "humor", "delight"],
    "annoyance": ["irritation", "bother", "displeasure", "vexation", "exasperation"],
    "approval": ["acceptance", "endorsement", "appreciation", "agreement", "praise"],
    "boredom": ["disinterest", "monotony", "dullness", "tedium", "ennui"],
    "caring": ["compassion", "concern", "kindness", "sympathy", "care"],
    "confusion": ["perplexity", "bewilderment", "puzzlement", "disorientation", "uncertainty", "confusion", "confused"],
    "curiosity": ["interest", "inquisitiveness", "wonder", "intrigue", "eagerness", "curious"],
    "desire": ["longing", "wish", "yearning", "craving", "want", "would"],
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
    "hate": ["loathing", "abhorrence", "detestation", "aversion", "hatred", "contempt", "revulsion", "disgust", "abhorring", "antipathy", "hostility", "enmity", "antagonism", "rage", "fury", "resentment", "spite", "bitterness", "hate"],
    "relief": ["ease", "comfort", "release", "freedom", "calm"],
    "remorse": ["guilt", "sorrow", "regret", "contrition", "penitence"],
    "neutral": ["indifferent", "neutral", "okay", "ok", "not bad", "okaish", "unbiased", "impartial", "apathetic", "disinterested", "detached", "unemotional", "lack of concern", "nonchalant", "unmoved", "calm", "composed"],
    "nervousness": ["anxiety", "apprehension", "unease", "restlessness", "fear"],
}

# Function to preprocess text (remove punctuation using NLTK)
def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')  # Matches only alphanumeric tokens
    tokens = tokenizer.tokenize(text)
    return " ".join(tokens)

# Function to detect language and translate if necessary
def detect_and_translate(text):
    try:
        detected_language = detect(text)
        if detected_language != 'en':
            translator = GoogleTranslator(source=detected_language, target='en')
            translated_text = translator.translate(text)
        else:
            translated_text = text
        return translated_text
    except LangDetectException as e:
        st.error(f"Language detection error: {e}")
        return text

# Function to handle negations in text
def handle_negations(text, original_emotion):
    negation_words = ["not", "no", "never", "don't", "isn't", "aren't", "won't", "can't", "didn't", "doesn't"]
    text_tokens = nltk.word_tokenize(text.lower())

    # Define negation-based emotion adjustments
    emotion_flip_map = {
        "sadness": "joy",
        "joy": "sadness",
        "love": "hate",
        "admiration": "disapproval",
        "approval": "disapproval",
        "anger": "calm",
        "fear": "confidence",
        "disgust": "admiration",
        "annoyance": "amusement",
        "optimism": "disappointment",
        "excitement": "fear",
    }

    # Check if any negation word is present in the tokens
    contains_negation = any(word in text_tokens for word in negation_words)

    # Adjust emotion if negation is detected
    if contains_negation and original_emotion in emotion_flip_map:
        return emotion_flip_map[original_emotion]

    return original_emotion

# Function to analyze emotion and sentiment
def analyze_emotion_and_sentiment(text):
    result = emotion_classifier(text)
    original_emotion = result[0]['label'] if result else "neutral"
    confidence = result[0]['score'] if result else 0

    # Check for synonyms in the text
    for emotion, synonyms in emotion_synonyms.items():
        if any(synonym in text.lower() for synonym in synonyms):
            original_emotion = emotion
            break

    # Handle negation and adjust emotion
    adjusted_emotion = handle_negations(text, original_emotion)

    # Define sentiment based on the adjusted emotion
    positive_emotions = ["admiration", "amusement", "approval", "joy", "love", "optimism", "pride", "relief", "gratitude", "excitement", "hope", "desire", "caring"]
    neutral_emotions = ["neutral", "boredom", "confusion", "curiosity", "nervousness", "realisation"]
    sentiment = "positive" if adjusted_emotion in positive_emotions else "neutral" if adjusted_emotion in neutral_emotions else "negative"
    return adjusted_emotion, sentiment, confidence

# Function to validate review input
def is_valid_review(review, min_length=10):
    if len(review.strip()) < min_length:
        return False
    if re.match(r'^[0-9]+$', review) or re.match(r'^[!@#$%^&*(),.?":{}|<>]+$', review):
        return False
    invalid_phrases = ["hi", "hello", "good morning", "hey", "howdy", "greetings", "good evening"]
    if review.strip().lower() in invalid_phrases:
        return False
    return True

# Streamlit App
st.set_page_config(page_title="Emotion-Driven Sentiment Analysis", layout="wide")

# Add custom CSS for sky-blue background, light blue input text, and sidebar background
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #F0F8FF;  /* Sky-blue color for the entire app */
    }

    /* Input text area styling */
    .stTextArea textarea, .stTextInput input {
        color: black;  /* Black text color */
        background-color: white;  /* White background */
        border: 2px solid #0000FF;  /* Blue border */
        border-radius: 5px;  /* Rounded corners */
        font-size: 16px;  /* Font size */
        padding: 10px;  /* Padding inside the input field */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.image("assets/AVN logo.jpg", width=800)
st.markdown("""
    <style>.text{font-weight:bold;font-size:40px}</style>
    <div class="text"><b>Welcome to Emotion-Driven Multilingual Sentiment Analysis Web Application</b></div>
    """, unsafe_allow_html=True)

# Add custom CSS for sidebar background
st.markdown(
    """
    <style>
    /* Target the sidebar container */
    [data-testid="stSidebar"] {
        background-color: #F0F8FF !important;  /* Alice Blue color for the sidebar */
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for Project Details
with st.sidebar:
    st.sidebar.image("assets/team.jpeg", width=400)
    st.markdown(
        """
        <h1>📋 Project Details</h1>
        <h2>🏫 AIML Department</h2>
        <p><strong>👨‍🏫 Department Head:</strong> Dr.M.Jayaram (Professor)</p>
        <p><strong>👩‍🏫 Project Guide:</strong> CH.Jyothi (Assistant Professor)</p>
        <p><strong>📅 Project Batch:</strong> A-12</p>
        <h3>👥 Team Members:</h3>
        <ul>
            <li>👩‍💻 K. Sri Ramya</li>
            <li>👩‍💻 Chikkapalli Lavanya</li>
            <li>👨‍💻 Endrapally Dinesh</li>
            <li>👨‍💻 Kompalli Mahesh</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

# User Guide in the main page as a dropdown option
with st.expander("📘 User Guide", expanded=False):
    st.markdown("""
        <h1>📘 User Guide</h1>
        <p>This app performs sentiment analysis on user input text or reviews in a CSV file. 
        It includes the following features:</p>
        <ul>
            <li><strong>📝 Text Sentiment Analysis:</strong> Detects the language, translates it (if needed), and predicts emotions.</li>
            <li><strong>📂 Bulk Analysis:</strong> Upload a CSV file containing reviews, and the app will predict emotions for each review.</li>
            <li><strong>📊 Emotion Count Summary:</strong> View the count of each emotion predicted for the uploaded reviews.</li>
            <li><strong>📥 Download Results:</strong> After analysis, you can download the results as a CSV file with predictions.</li>
        </ul>
        <h3>📝 Example:</h3>
        <p><strong>Input (Hindi):</strong> "यह फिल्म बहुत शानदार थी, मुझे इसकी कहानी और अभिनय बहुत पसंद आया।"</p>
        <p><strong>Predicted Emotion:</strong> ❤️ Love</p>
    """, unsafe_allow_html=True)

# Single Review Analysis
st.subheader("📝 Analyze a Single Review")
user_input = st.text_area("Enter a review:")

if st.button("Analyze Review 🚀"):
    if user_input.strip():
        if is_valid_review(user_input):
            # Preprocess text (remove punctuation)
            user_input = preprocess_text(user_input)
            translated_text = detect_and_translate(user_input)
            emotion, sentiment, confidence = analyze_emotion_and_sentiment(translated_text)

            st.success(f"🎭 Emotion: {emotion.capitalize()} (Confidence: {confidence:.2f})")
            st.success(f"📊 Sentiment: {sentiment.capitalize()}")

            # Display corresponding image for emotion
            if emotion in emotion_images:
                st.image(emotion_images[emotion], caption=emotion.capitalize())
        else:
            st.error(f"❌ Invalid review. Please provide a valid review.")
    else:
        st.warning("⚠️ Please enter a review.")

# Bulk Analysis
st.subheader("📂 Bulk Analysis of Reviews")
uploaded_file = st.file_uploader("Upload a CSV file with a text column 📄", type=["csv"])

if uploaded_file:
    try:
        # Read the file as bytes and detect encoding using chardet
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # Create a BytesIO object to use it like a file
        raw_data_io = BytesIO(raw_data)
        df = pd.read_csv(raw_data_io, encoding=encoding)
        
    except Exception as e:
        st.error(f"❌ Error reading the uploaded file: {e}")
        st.stop()

    # Dynamically choose the review column
    review_columns = [
        col for col in df.columns
        if any(
            keyword in col.lower()
            for keyword in ['review', 'feedback', 'text', 'texts', 'reviews', 'feedbacks']
        )
    ]

    if review_columns:
        text_column = st.selectbox("Select the Review Column 📝", review_columns)
    else:
        st.warning("⚠️ No appropriate review column found. Please check your dataset.")
        st.stop()

    if text_column:
        # Validate if the column contains strings (reviews)
        valid_rows = df[text_column].apply(lambda x: isinstance(x, str))
        invalid_rows_count = (~valid_rows).sum()

        if invalid_rows_count > 0:
            st.warning(
                f"⚠️ The column '{text_column}' contains {invalid_rows_count} invalid entries. "
                f"These will be ignored in the analysis."
            )

        # Filter only valid rows
        valid_reviews = df[text_column][valid_rows].fillna("").tolist()
        emotions, sentiments, validity_flags = [], [], []

        for review in valid_reviews:
            try:
                if is_valid_review(review):  # Use the is_valid_review function here
                    # Preprocess text (remove punctuation)
                    review = preprocess_text(review)
                    translated_text = detect_and_translate(review)  # Detect language and translate
                    emotion, sentiment, _ = analyze_emotion_and_sentiment(translated_text)  # Analyze after translation
                    emotions.append(emotion)
                    sentiments.append(sentiment)
                    validity_flags.append("valid")
                else:
                    emotions.append("invalid")
                    sentiments.append("invalid")
                    validity_flags.append("invalid")
            except Exception as e:
                # Handle errors for individual reviews
                st.error(f"❌ Error processing review: '{review}'. Error: {e}")
                emotions.append("error")
                sentiments.append("error")
                validity_flags.append("error")

        # Add results to the dataframe for valid rows
        valid_df = df.loc[valid_rows].copy()
        valid_df['Emotion'] = emotions
        valid_df['Sentiment'] = sentiments
        valid_df['Validity'] = validity_flags

        # Separate valid and invalid reviews
        valid_reviews_df = valid_df[valid_df['Validity'] == 'valid']
        invalid_reviews_df = valid_df[valid_df['Validity'] != 'valid']

        # Summarize emotions and sentiments
        emotion_summary = valid_reviews_df['Emotion'].value_counts().reset_index()
        emotion_summary.columns = ['Emotion', 'Count']  # Rename columns explicitly

        sentiment_summary = valid_reviews_df['Sentiment'].value_counts().reset_index()
        sentiment_summary.columns = ['Sentiment', 'Count']  # Rename columns explicitly

        # Display results
        st.write("### 📊 Valid Reviews Analysis")
        st.dataframe(valid_reviews_df[[text_column, 'Emotion', 'Sentiment']])

        st.write("### 📈 Emotion Count Summary")
        st.dataframe(emotion_summary)

        st.write("### 📉 Sentiment Count Summary")
        st.dataframe(sentiment_summary)

        # Display invalid reviews
        st.write("### ❌ Invalid Reviews")
        st.dataframe(invalid_reviews_df[[text_column, 'Validity']])

        # Download button for processed results
        processed_csv = valid_reviews_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Processed Valid Results", processed_csv, "processed_valid_reviews.csv", "text/csv")

        invalid_csv = invalid_reviews_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Invalid Reviews", invalid_csv, "invalid_reviews.csv", "text/csv")
