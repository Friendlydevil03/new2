from flask import Flask, request, render_template
import numpy as np
from feature import FeatureExtraction
import os

app = Flask(__name__)


def simple_url_check(features):
    # Convert features to list if they're numpy array
    features = features.ravel() if isinstance(features, np.ndarray) else features

    # Simple rules (example threshold-based decision)
    suspicious_count = sum(1 for f in features if abs(f) > 0.5)
    is_suspicious = suspicious_count > len(features) // 2

    # Calculate confidence - higher for more definitive results
    confidence_ratio = suspicious_count / len(features)
    confidence = confidence_ratio if is_suspicious else (1 - confidence_ratio)

    # Return 0 for suspicious URLs and 1 for safe ones, with appropriate confidence
    return (0 if is_suspicious else 1, confidence)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        # Use simple rule-based prediction instead of untrained classifier
        prediction, confidence = simple_url_check(x)

        return render_template('index.html',
                               xx=round(confidence, 2),
                               url=url,
                               prediction=prediction)
    return render_template("index.html", xx=-1)


@app.route("/admin/train", methods=["GET", "POST"])
def admin_train():
    """Admin page for model training"""
    global model

    if request.method == "POST":
        if 'dataset_file' not in request.files:
            return render_template('admin.html', error="No file selected")

        file = request.files['dataset_file']
        if file.filename == '':
            return render_template('admin.html', error="No file selected")

        if file and file.filename.endswith('.csv'):
            # Save uploaded file
            dataset_path = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(dataset_path)

            try:
                # Import here to avoid circular imports
                from train_model import train_model
                new_model, accuracy = train_model(dataset_path)

                if new_model is not None:
                    model = new_model  # Update the model in the app
                    return render_template('admin.html',
                                           success=f"Model trained successfully with {accuracy:.2%} accuracy")
                else:
                    return render_template('admin.html',
                                           error="Training failed. Check logs for details.")
            except Exception as e:
                return render_template('admin.html',
                                       error=f"Error during training: {str(e)}")

    return render_template('admin.html')

if __name__ == "__main__":
    app.run(debug=True)