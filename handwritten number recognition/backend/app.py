from flask import Flask, request, jsonify, render_template
from predict import predict_digit
import os

app = Flask(__name__)

# Serve the frontend
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint for digit prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        predicted_digit = predict_digit(file)
        return jsonify({"prediction": int(predicted_digit)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
