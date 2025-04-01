from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import os
import logging
from datetime import datetime
import secrets
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from werkzeug.middleware.proxy_fix import ProxyFix
import ipaddress
import re
import urllib.request
from bs4 import BeautifulSoup
import socket
import requests
from googlesearch import search
import whois
from datetime import date, datetime
import time
from dateutil.parser import parse as date_parse
from urllib.parse import urlparse
import logging

# Setup logging
logger = logging.getLogger(__name__)


class FeatureExtraction:
    """
    Class to extract features from URLs for phishing detection
    """

    def __init__(self, url):
        self.features = []
        self.url = url
        self.domain = ""
        self.whois_response = None
        self.urlparse = None
        self.response = None
        self.soup = None

        # Initialize URL parsing
        try:
            self.urlparse = urlparse(url)
            self.domain = self.urlparse.netloc
        except Exception as e:
            logger.error(f"Error parsing URL: {e}")

        # Get response and parse HTML
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
            }
            self.response = requests.get(url, headers=headers, timeout=5)
            self.soup = BeautifulSoup(self.response.text, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching URL content: {e}")

        # Get WHOIS information
        try:
            self.whois_response = whois.whois(self.domain)
        except Exception as e:
            logger.error(f"Error getting WHOIS data: {e}")

        # Extract features - no longer calling a method that doesn't exist
        self._extract_features()

    def _extract_features(self):
        """Extract all features from the URL"""
        # Initialize with some default values for all 30 features
        self.features = [0] * 30

        # Extract some basic features

        # 1. Length of URL
        self.features[0] = len(self.url)

        # 2. Number of dots in domain
        self.features[1] = self.domain.count('.')

        # 3. URL contains IP address
        self.features[2] = 1 if self._has_ip_address() else 0

        # 4. URL length is suspicious
        self.features[3] = 1 if len(self.url) > 75 else 0

        # 5. URL contains '@' symbol
        self.features[4] = 1 if '@' in self.url else 0

        # 6. URL contains double slash redirection
        self.features[5] = 1 if '//' in self.url[8:] else 0

        # 7. URL has prefix/suffix with hyphen
        self.features[6] = 1 if '-' in self.domain else 0

        # 8. Domain contains subdomain
        self.features[7] = 1 if self.domain.count('.') > 1 else 0

        # 9. Domain uses HTTPS
        self.features[8] = 1 if self.url.startswith('https') else 0

        # 10. Domain registration length
        try:
            if self.whois_response and self.whois_response.expiration_date:
                expiration_date = self.whois_response.expiration_date
                if isinstance(expiration_date, list):
                    expiration_date = expiration_date[0]
                current_date = datetime.now()
                registration_length = (expiration_date - current_date).days
                self.features[9] = 1 if registration_length > 365 else 0
            else:
                self.features[9] = 0
        except:
            self.features[9] = 0

        # Fill the rest of the features with default values (0)
        # In a real implementation, you would calculate all 30 features

    def _has_ip_address(self):
        """Check if the URL contains an IP address"""
        pattern = re.compile(
            r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5]))')
        return pattern.search(self.url) is not None

    def getFeaturesList(self):
        """Return the list of features extracted from the URL"""
        return self.features


# The rest of your feature.py file (model loading, app routes, etc.) remains unchanged
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
app.config['SECRET_KEY'] = secrets.token_hex(16)

# Set up model versioning
MODEL_PATH = 'models/url_classifier_model.pkl'
MODEL_VERSION_PATH = 'models/model_version.txt'


def load_model():
    """Load the ML model or train a new one if no model file exists"""
    os.makedirs('models', exist_ok=True)

    try:
        # Try to load an existing model
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)

        # Get model version
        try:
            with open(MODEL_VERSION_PATH, 'r') as version_file:
                version = version_file.read().strip()
        except FileNotFoundError:
            version = "unknown"

        logger.info(f"Loaded existing model (version: {version})")
        return model
    except FileNotFoundError:
        logger.warning("No model found. Creating a new placeholder model...")
        # Create a new model if no saved model exists
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )

        # Save model version (initial)
        with open(MODEL_VERSION_PATH, 'w') as version_file:
            version = datetime.now().strftime("%Y%m%d%H%M%S")
            version_file.write(version)

        # Save the model
        with open(MODEL_PATH, 'wb') as model_file:
            pickle.dump(model, model_file)

        return model


# Load the model when the application starts
model = load_model()


@lru_cache(maxsize=100)
def predict_url(url):
    """
    Make a prediction for a URL with caching for performance

    Args:
        url (str): The URL to analyze

    Returns:
        tuple: (prediction, confidence)
    """
    try:
        # Extract features
        feature_extractor = FeatureExtraction(url)
        features = np.array(feature_extractor.getFeaturesList()).reshape(1, 30)

        # Check if model needs to be trained
        if not hasattr(model, 'classes_'):
            logger.warning("Model not trained. Using random prediction.")
            prediction = np.random.randint(0, 2)
            confidence = np.random.random()
        else:
            # Use the trained model to make a prediction
            prediction = model.predict(features)[0]
            confidence_scores = model.predict_proba(features)[0]
            confidence = confidence_scores[1] if prediction == 1 else confidence_scores[0]

        return int(prediction), float(confidence)
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        # Return a safe default (mark as suspicious) in case of any error
        return 0, 0.5


@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for the web interface"""
    if request.method == "POST":
        url = request.form.get("url", "")
        if not url:
            return render_template('index.html', xx=-1, error="Please enter a URL")

        # Log the URL being checked (for monitoring purposes)
        logger.info(f"Checking URL: {url}")

        # Get prediction
        prediction, confidence = predict_url(url)

        return render_template('index.html',
                               xx=round(confidence, 2),
                               url=url,
                               prediction=prediction)

    return render_template("index.html", xx=-1)


@app.route("/api/check", methods=["POST"])
def api_check():
    """API endpoint for programmatic access"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing URL parameter"}), 400

    url = data['url']
    prediction, confidence = predict_url(url)

    return jsonify({
        "url": url,
        "is_safe": bool(prediction),
        "confidence": round(confidence, 4),
        "timestamp": datetime.now().isoformat()
    })


@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    API endpoint to train the model
    Expects training data in JSON format with features and labels
    """
    global model

    # In a production system, you would want to authenticate this endpoint
    try:
        data = request.get_json()
        if not data or 'features' not in data or 'labels' not in data:
            return jsonify({"error": "Missing training data"}), 400

        X_train = np.array(data['features'])
        y_train = np.array(data['labels'])

        # Train the model
        model.fit(X_train, y_train)

        # Save the model
        os.makedirs('models', exist_ok=True)
        with open(MODEL_PATH, 'wb') as model_file:
            pickle.dump(model, model_file)

        # Update version
        with open(MODEL_VERSION_PATH, 'w') as version_file:
            version = datetime.now().strftime("%Y%m%d%H%M%S")
            version_file.write(version)

        # Clear the prediction cache since we have a new model
        predict_url.cache_clear()

        return jsonify({
            "status": "success",
            "message": "Model trained and saved successfully",
            "version": version
        })
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "ok",
        "model_ready": hasattr(model, 'classes_'),
        "timestamp": datetime.now().isoformat()
    })


# Error handler for 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404


# Error handler for 500 errors
@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error="Internal server error"), 500


if __name__ == "__main__":
    # In production, you would use a proper WSGI server like Gunicorn
    # and set debug=False
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)