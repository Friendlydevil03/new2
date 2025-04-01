import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import logging
from datetime import datetime
from tqdm import tqdm
import argparse
from feature import FeatureExtraction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def extract_features_from_url(url):
    """Extract features from a single URL"""
    try:
        extractor = FeatureExtraction(url)
        features = extractor.getFeaturesList()
        return features
    except Exception as e:
        logger.error(f"Error extracting features for URL {url}: {str(e)}")
        return None


def train_model(dataset_path, test_size=0.2, n_estimators=100, random_state=42):
    """
    Train a Random Forest model using the provided dataset

    Args:
        dataset_path: Path to CSV file with 'url' and 'label' columns
        test_size: Proportion of data to use for testing
        n_estimators: Number of trees in the Random Forest
        random_state: Random seed for reproducibility

    Returns:
        Trained model and performance metrics
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

    # Check if dataset has required columns
    required_columns = ['url', 'label']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Dataset missing required columns. Found: {df.columns.tolist()}, Required: {required_columns}")
        return None

    # Extract features
    logger.info("Extracting features from URLs...")
    features_list = []
    labels = []

    # Use tqdm for progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        url = row['url']
        label = row['label']

        features = extract_features_from_url(url)
        if features is not None:
            features_list.append(features)
            labels.append(label)

    # Convert to numpy arrays
    X = np.array(features_list)
    y = np.array(labels)

    logger.info(f"Feature extraction complete. Obtained {len(X)} feature vectors.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train model
    logger.info(f"Training Random Forest model with {n_estimators} estimators...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )

    model.fit(X_train, y_train)

    # Evaluate model
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Model trained. Accuracy on test set: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': range(X.shape[1]),
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    logger.info("\nTop 10 Important Features:")
    logger.info(feature_importance.head(10))

    # Save model
    model_path = 'models/url_classifier_model.pkl'
    logger.info(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save version info
    version = datetime.now().strftime("%Y%m%d%H%M%S")
    with open('models/model_version.txt', 'w') as f:
        f.write(f"{version}_samples{len(X)}_acc{accuracy:.4f}")

    logger.info("Model training complete!")
    return model, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train phishing URL detection model")
    parser.add_argument("--dataset", required=True, help="Path to the dataset CSV file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data for testing")
    parser.add_argument("--estimators", type=int, default=100, help="Number of trees in Random Forest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset,
        test_size=args.test_size,
        n_estimators=args.estimators,
        random_state=args.seed
    )