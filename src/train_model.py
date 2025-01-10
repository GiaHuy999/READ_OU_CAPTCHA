import os
import cv2
import csv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from keras.layers import Reshape, Dense

def load_data(data_dir, labels_file):
    with open(labels_file, 'r') as f:
        reader = csv.DictReader(f)
        labels_dict = {row['file_name']: row['label'] for row in reader}

    X, y = [], []
    for file_name, label in labels_dict.items():
        img_path = os.path.join(data_dir, file_name)
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} does not exist.")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read {img_path}.")
            continue

        img = cv2.resize(img, (100, 40))
        img = img.reshape((40, 100, 1)) / 255.0
        X.append(img)

        numeric_label = []
        for c in label:
            if '0' <= c <= '9':
                numeric_label.append(ord(c) - ord('0'))
            elif 'A' <= c <= 'Z':
                numeric_label.append(ord(c) - ord('A') + 10)
            elif 'a' <= c <= 'z':
                numeric_label.append(ord(c) - ord('a') + 36)
            else:
                raise ValueError(f"Unsupported character in label: {c}")
        
        y.append(numeric_label)

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(X)} samples.")
    return X, y

def train_model(X, y):
    num_classes = np.max(y) + 1
    num_characters_per_captcha = y.shape[1]

    y = np.array([to_categorical(label, num_classes).reshape(-1) for label in y])

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(40, 100, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_characters_per_captcha * num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    model.add(Dense(4 * 62))  
    model.add(Reshape((4, 62))) 
    return model

if __name__ == "__main__":
    processed_captchas_dir = "./data"
    labels_csv = "./labels.csv"
    
    X, y = load_data(processed_captchas_dir, labels_csv)

    model = train_model(X, y)

    model.save("captcha_reader_model.h5")
    print("Model saved as captcha_reader_model.h5")
