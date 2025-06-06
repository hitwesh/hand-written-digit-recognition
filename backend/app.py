import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../frontend/templates"))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "../frontend/static"))

print(f"Template folder path: {TEMPLATE_DIR}")
print(f"Static folder path: {STATIC_DIR}")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

#for defning cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# to load all the models :3
device = torch.device("cpu")
cnn_model = CNN()
cnn_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "../models/cnn_model.pth"), map_location=device))
cnn_model.eval()

svm_model = joblib.load(os.path.join(BASE_DIR, "../models/svm_model.pkl"))
rfc_model = joblib.load(os.path.join(BASE_DIR, "../models/rfc_model.pkl"))
knn_model = joblib.load(os.path.join(BASE_DIR, "../models/knn_model.pkl"))

# route for frontend
@app.route("/")
def home():
    try:
        print(f"Attempting to render template from: {TEMPLATE_DIR}")
        return render_template("index.html")
    except Exception as e:
        print(f"Error rendering template: {e}, Template folder: {TEMPLATE_DIR}")
        return "Error rendering template", 500

# Preprocess imggg
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)
    image = (image / 255.0).astype(np.float32)  # Normalize
    return image

# Prediction route :0
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file)

        img_array = preprocess_image(image)

        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        img_flat = img_array.flatten().reshape(1, -1)

        cnn_pred = torch.argmax(cnn_model(img_tensor)).item()
        svm_pred = int(svm_model.predict(img_flat)[0])
        rfc_pred = int(rfc_model.predict(img_flat)[0])
        knn_pred = int(knn_model.predict(img_flat)[0])

        return jsonify({
            "cnn_prediction": cnn_pred,
            "svm_prediction": svm_pred,
            "rfc_prediction": rfc_pred,
            "knn_prediction": knn_pred
        })
    
    except Exception as e:
        return jsonify({"error": "An error occurred during prediction: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
