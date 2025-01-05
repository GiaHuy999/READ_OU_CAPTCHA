import os
import cv2
import csv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Step 1: Load and prepare data
def load_data(data_dir, labels_file):
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        labels_dict = {row['file_name']: row['label'] for row in reader}

    X, y = [], []
    for file_name, label in labels_dict.items():
        img = cv2.imread(os.path.join(data_dir, file_name), cv2.IMREAD_GRAYSCALE)
        img = img.reshape((40, 100, 1)) / 255.0  # Normalize
        X.append(img)
        
        # Chuyển nhãn thành dạng số
        y.append([ord(c) - ord('0') if c.isdigit() else ord(c) - ord('A') + 10 for c in label])

    return np.array(X), np.array(y)

# Step 2: Build and train the model
def train_model(X, y, num_classes=62):  # Chỉnh sửa num_classes để bao gồm cả chữ cái và số
    y = [to_categorical(label, num_classes) for label in y]
    y = np.array(y)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(40, 100, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes * len(y[0]), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    return model

# Main execution
if __name__ == "__main__":
    processed_captchas_dir = "D:/OU/Captcha/data/processed"
    labels_csv = "D:/OU/Captcha/data/labels.csv"

    # Step 1: Load data
    X, y = load_data(processed_captchas_dir, labels_csv)

    # Step 2: Train model
    model = train_model(X, y)

    # Save the model
    model.save("captcha_reader_model.h5")
    print("Model saved as captcha_reader_model.h5")