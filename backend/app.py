from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the sentiment analysis pipeline
sentiment_model = pipeline('sentiment-analysis')

# Endpoint for sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data['text']

    # Perform sentiment analysis
    result = sentiment_model(text)[0]
    label = result['label']

    # Determine if the text is pro-vegan or anti-vegan
    if label == 'POSITIVE':
        sentiment = 'Pro-Vegan'
    else:
        sentiment = 'Anti-Vegan'

    return jsonify({'result': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
