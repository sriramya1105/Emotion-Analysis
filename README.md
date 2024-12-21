# Sentiment_analyser

### Dataset Description for Twitter Emotion Prediction:

- *Dataset Name*: English Twitter Messages with Six Basic Emotions
- *Total Records*: 2,000 tweets
- *File Format*: CSV (Comma-separated values)

#### Columns:
1. *Text (Tweet)*:
   - *Type*: String
   - *Description*: Contains the text of the tweet, which is written in English.
   - *Example*: "I am feeling so joyful today, the world seems brighter!"

2. *Label*:
   - *Type*: Integer
   - *Description*: Encoded emotion label for each tweet. The emotion labels are already encoded as integers.
   - *Possible Values*:
     - 0: *Sadness* – Tweets expressing sadness or grief.
     - 1: *Joy* – Tweets expressing happiness or joy.
     - 2: *Love* – Tweets expressing affection or love.
     - 3: *Anger* – Tweets expressing frustration or anger.
     - 4: *Fear* – Tweets expressing fear or anxiety.
     - 5: *Surprise* – Tweets expressing shock or surprise.

#### Example Data:
| Text                                             | Label |
|--------------------------------------------------|-------|
| "I'm so sad today. Nothing seems to work out."    | 0     |
| "Wow! I just won the lottery, this is amazing!"   | 1     |
| "I love spending time with my family!"           | 2     |
| "This is so frustrating, I can't take it anymore!"| 3     |
| "I'm really scared about the upcoming exam."      | 4     |
| "What just happened? I can't believe it!"         | 5     |

#### Dataset Characteristics:
- *Text Data*: Each tweet is short, typically under 280 characters, as per Twitter's limit.
- *Emotional Labels*: Each tweet is assigned one of six basic emotions, encoded as integers.
- *Balanced Classes*: Check for class distribution to see if some emotions are more frequent than others.

#### Usage:
- *Task*: Emotion classification – training a model to predict the emotional tone of a tweet.
- *Model Input*: Preprocessed tweet text.
- *Model Output*: Predicted emotion label (one of 0-5).

This dataset can be used for various Natural Language Processing (NLP) tasks, such as text classification, sentiment analysis, and emotion prediction from social media content.
