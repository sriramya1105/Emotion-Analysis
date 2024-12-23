import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect
from deep_translator import GoogleTranslator
from PIL import Image

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to load and resize images
def load_and_resize_image(image_path, size=(300, 300)):
    img = Image.open(image_path)
    img = img.resize(size)
    return img

# Function for text preprocessing
def lemmatizing(content):
    content = re.sub(r'http\S+|www\S+|https\S+', '', content, flags=re.MULTILINE)
    content = re.sub(r'@\w+', '', content)
    content = re.sub('[^a-zA-Z]', ' ', content)

    content = content.lower()
    content = content.split()
    content = [lemmatizer.lemmatize(word) for word in content if word not in stop_words]
    return ' '.join(content)

# Function for language detection and translation
def detect_and_translate(text):
    # Detect language
    detected_language = detect(text)   
    # Translate to English if not already in English
    if detected_language != 'en':
        translator = GoogleTranslator(source=detected_language, target='en')
        translated_text = translator.translate(text)
    else:
        translated_text = text

    return detected_language, translated_text

# Map emotions to sentiments
emotion_to_sentiment = {
    "joy": "positive",
    "love": "positive",
    "admiration": "positive",
    "amusement": "positive",
    "approval": "positive",
    "caring": "positive",
    "desire": "positive",
    "excitement": "positive",
    "gratitude": "positive",
    "optimism": "positive",
    "pride": "positive",
    "relief": "positive",
    "neutral": "neutral",
    "realisation": "neutral",
    "surprise": "neutral",
    "anger": "negative",
    "annoyance": "negative",
    "disappointment": "negative",
    "disapproval": "negative",
    "disgust": "negative",
    "fear": "negative",
    "grief": "negative",
    "nervousness": "negative",
    "remorse": "negative",
    "sadness": "negative",
    "envy": "negative",
    "boredom": "negative",
    "confusion": "negative",
    "embarrassment": "negative"
}

# Load dataset for training the model
try:
    data = pd.read_csv("Emotions(2).csv")  # Ensure the file exists in the directory

    # Preprocess the data
    data['Reviews'] = data['Reviews'].apply(lemmatizing)

    # Split data into features and labels
    X = data['Reviews']
    y = data['Emotion']

    # Convert text data to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=2000, solver='liblinear')
    model.fit(X_train, y_train)
except FileNotFoundError:
    st.error("The file Emotions.csv is missing. Please ensure it is in the same directory as this script.")
    model = None
    vectorizer = None

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
- **Input (Telugu)**: "నేను చాలా సంతోషంగా ఉన్నాను"
- **Detected Language**: Telugu
- **Translated Text**: "I am very happy"
- **Predicted Emotion**: Joy

In this example, the text is in Telugu, and the app detects the language, translates it into English, and then predicts the emotion as "Joy."
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

# User Input Text Analysis
st.subheader("Test Your Text")
user_input = st.text_area("Enter text to classify:")
if st.button("Predict Emotion") and model and vectorizer:
    if user_input.strip():
        # Detect language and translate
        detected_language, translated_text = detect_and_translate(user_input)

        # Display detected language and translated text
        st.write(f"Translated Text: {translated_text}")

        # Preprocess and vectorize the input
        processed_input = lemmatizing(translated_text)
        input_vector = vectorizer.transform([processed_input])

        # Make prediction
        prediction = model.predict(input_vector)[0]
        predicted_sentiment = emotion_to_sentiment.get(prediction.lower(), "neutral")

        # Display predicted emotion
        st.success(f"Predicted Emotion: {prediction}")

        image_mapping = {
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

        # Use the prediction as the key for the image mapping
        if prediction.lower() in image_mapping:
            img = load_and_resize_image(image_mapping[prediction.lower()], size=(300, 300))
            st.image(img, caption=prediction.capitalize())
        else:
            st.warning("No image available for this emotion.")

        # Display emoji feedback based on sentiment
        if predicted_sentiment == "positive":
            st.markdown("😊 **Overall Sentiment is Positive!**")
        elif predicted_sentiment == "negative":
            st.markdown("😢 **Overall Sentiment is Negative!**")
        else:
            st.markdown("😐 **Overall Sentiment is Neutral.**")
    else:
        st.error("Please enter valid text.")

# Upload and process CSV file
st.subheader("Bulk Analysis with CSV File")
uploaded_file = st.file_uploader("Upload a CSV file for bulk analysis", type=["csv"])

if uploaded_file is not None and model and vectorizer:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
        column_name = st.selectbox("Select the column containing reviews:", options=uploaded_data.columns)

        if column_name:
            uploaded_data['Processed_Reviews'] = uploaded_data[column_name].astype(str).apply(lemmatizing)
            predictions = model.predict(vectorizer.transform(uploaded_data['Processed_Reviews']))
            uploaded_data['Predicted_Emotion'] = predictions
            uploaded_data['Predicted_Sentiment'] = uploaded_data['Predicted_Emotion'].apply(
                lambda emotion: emotion_to_sentiment.get(emotion.lower(), "neutral")
            )
            result_df = uploaded_data[[column_name, 'Predicted_Emotion', 'Predicted_Sentiment']]
            st.write(result_df)
            emotion_counts = uploaded_data['Predicted_Emotion'].value_counts()
            sentiment_counts = uploaded_data['Predicted_Sentiment'].value_counts()
            st.subheader("Emotion Count Summary")
            st.write(emotion_counts)
            st.subheader("Sentiment Count Summary")
            st.write(sentiment_counts)

            # Display dominant sentiment emoji
            dominant_sentiment = sentiment_counts.idxmax()
            emoji_mapping = {
                "positive": "assets/joy.jpg",
                "negative": "assets/sad.jpg",
                "neutral": "assets/neutral.jpeg"
            }

            if dominant_sentiment in emoji_mapping:
                emoji_path = emoji_mapping[dominant_sentiment]
                emoji_image = load_and_resize_image(emoji_path, size=(150, 150))
                st.image(emoji_image, caption=f"Dominant Sentiment: {dominant_sentiment.capitalize()}")
            else:
                st.warning("No emoji available for the dominant sentiment.")

            csv = result_df.to_csv(index=False)
            st.download_button(label="Download Results", data=csv, file_name="analysis_results.csv", mime="text/csv")
        else:
            st.error("Please select a valid column containing review text.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

elif not model or not vectorizer:
    st.error("The model could not be loaded. Ensure the Emotions.csv file is present for training.")

else:
    st.info("Please upload a CSV file to proceed.")
