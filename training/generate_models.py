import os
import joblib
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Create model dir if it doesnt exists
if not os.path.exists("models"):
    os.makedirs("models")

# mnsit loader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# trains only 10k, im broke no good pc
X_train = trainset.data[:10000].numpy().reshape(10000, -1)  # (100, 28*28)
y_train = trainset.targets[:10000].numpy()

print(f"✅ Using {len(X_train)} samples for training...")

# Train Support Vector Machine (SVM)
print("🔄 Training SVM... (This will be fast)")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "models/svm_model.pkl")
print("✅ SVM trained and saved!")

# Train Random Forest Classifier (RFC)
print("🔄 Training Random Forest... (This will be fast)")
rfc_model = RandomForestClassifier(n_estimators=10, random_state=42)  # Reduced estimators for speed
rfc_model.fit(X_train, y_train)
joblib.dump(rfc_model, "models/rfc_model.pkl")
print("✅ Random Forest trained and saved!")

# Train K-Nearest Neighbors (KNN)
print("🔄 Training KNN... (This will be fast)")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
joblib.dump(knn_model, "models/knn_model.pkl")
print("✅ KNN trained and saved!")

print("🎉 All models trained successfully (using only 10000 samples)!")
