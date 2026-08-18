from flask import Flask, render_template, request, jsonify

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


app = Flask(__name__)


# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("dataset.csv")

texts = data["text"].values
emotions = data["emotion"].values


# ==========================================
# CONVERT TEXT INTO NUMBERS
# ==========================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts).toarray()


# ==========================================
# CREATE EMOTION LABELS
# ==========================================

emotion_names = sorted(list(set(emotions)))

emotion_to_number = {
    emotion: number
    for number, emotion in enumerate(emotion_names)
}


y = np.array([
    emotion_to_number[emotion]
    for emotion in emotions
])


# ==========================================
# DEEP LEARNING MODEL
# ==========================================

model = Sequential([

    Dense(
        64,
        activation="relu",
        input_shape=(X.shape[1],)
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        len(emotion_names),
        activation="softmax"
    )

])


model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Emotion Detection Model...\n")

model.fit(

    X,
    y,

    epochs=100,

    verbose=0

)

print("====================================")
print("MODEL TRAINED SUCCESSFULLY!")
print("====================================")


# ==========================================
# EMOTION PREDICTION
# ==========================================

def predict_emotion(text):

    vector = vectorizer.transform(
        [text]
    ).toarray()

    prediction = model.predict(
        vector,
        verbose=0
    )[0]

    index = np.argmax(prediction)

    confidence = prediction[index]

    emotion = emotion_names[index]

    return emotion, confidence


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# PREDICTION API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    text = data.get("text", "")

    if text.strip() == "":

        return jsonify({
            "emotion": "Please enter some text."
        })


    emotion, confidence = predict_emotion(text)


    return jsonify({

        "emotion": emotion,

        "confidence": round(
            float(confidence) * 100,
            2
        )

    })


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)