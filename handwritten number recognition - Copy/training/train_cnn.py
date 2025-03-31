import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Check if GPU is available
device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f"🚀 Training on: {device_name}")

# Create 'models' directory if it doesn't exist
if not os.path.exists("models"):
    os.makedirs("models")

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize images (scale pixel values between 0 and 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape data to fit CNN input shape (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("🔄 Training CNN... (This may take a few minutes)")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Save the trained model
model.save("models/cnn_model.h5")
print("✅ CNN trained and saved as 'models/cnn_model.h5'!")

# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"📊 Model Accuracy: {accuracy:.4f}")
