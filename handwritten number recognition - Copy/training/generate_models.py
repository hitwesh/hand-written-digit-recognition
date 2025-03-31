import os
import joblib
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create 'models' directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Use the FULL dataset (60,000 samples)
X_train = trainset.data.numpy().reshape(len(trainset), -1)  # (60000, 28*28)
y_train = trainset.targets.numpy()

print(f"✅ Using {len(X_train)} samples for training...")

# Train Support Vector Machine (SVM)
print("🔄 Training SVM... (This may take a few minutes)")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "models/svm_model.pkl")
print("✅ SVM trained and saved!")

# Train Random Forest Classifier (RFC)
print("🔄 Training Random Forest... (This may take a few minutes)")
rfc_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Increased estimators for better accuracy
rfc_model.fit(X_train, y_train)
joblib.dump(rfc_model, "models/rfc_model.pkl")
print("✅ Random Forest trained and saved!")

# Train K-Nearest Neighbors (KNN)
print("🔄 Training KNN... (This may take a few minutes)")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
joblib.dump(knn_model, "models/knn_model.pkl")
print("✅ KNN trained and saved!")

print("🎉 All models trained successfully (using 60,000 samples)!")
