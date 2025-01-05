import os
import cv2
import csv
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
from io import BytesIO

# Step 1: Collect and preprocess CAPTCHA images directly from the web
def collect_and_preprocess_captchas(url, num_captchas, save_dir):
    # Khởi tạo đối tượng ChromeOptions để cấu hình tùy chọn
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Chạy không cửa sổ trình duyệt

    # Khởi tạo WebDriver với các tùy chọn
    service = Service('D:/OU/Captcha/chromedriver-win64/chromedriver.exe')  # Cập nhật với đường dẫn chính xác
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_captchas):
        captcha_element = driver.find_element(By.TAG_NAME, 'img')
        captcha_src = captcha_element.get_attribute('src')
        
        # Tải ảnh CAPTCHA trực tiếp từ URL
        captcha_data = requests.get(captcha_src).content
        
        # Đọc ảnh từ byte stream và tiền xử lý
        img = cv2.imdecode(np.frombuffer(captcha_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Tiền xử lý ảnh
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 40))
        _, binarized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

        # Lưu ảnh đã xử lý
        file_path = os.path.join(save_dir, f'captcha_{i}.png')
        cv2.imwrite(file_path, binarized)
        print(f"Processed and saved CAPTCHA {i} to {file_path}")
        
        driver.refresh()
        time.sleep(0.1)

    driver.quit()

# Step 2: Label the CAPTCHA images
def label_captchas(output_dir, labels_file):
    # Xác định các ký tự hợp lệ
    valid_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    with open(labels_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_name", "label"])

        for file_name in os.listdir(output_dir):
            if file_name.endswith('.png'):
                print(f"Displaying: {file_name}")
                
                # Yêu cầu nhập nhãn
                label = input("Enter the CAPTCHA label: ")

                # Kiểm tra nếu nhãn chứa chỉ các ký tự hợp lệ
                while not all(c in valid_characters for c in label):
                    print(f"Invalid label: {label}. Only characters from {valid_characters} are allowed.")
                    label = input("Enter the CAPTCHA label again: ")

                # Ghi nhãn hợp lệ vào file CSV
                writer.writerow([file_name, label])
                print(f"Label for {file_name}: {label}")

# Step 3: Build and train the model
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
    captcha_url = "https://id.ou.edu.vn/captcha"
    processed_captchas_dir = "D:/OU/Captcha/data/processed"
    labels_csv = "D:/OU/Captcha/data/labels.csv"

    # Step 1: Collect and preprocess CAPTCHA images directly from the web
    collect_and_preprocess_captchas(captcha_url, num_captchas=3, save_dir=processed_captchas_dir)

    # Step 2: Label CAPTCHA images
    label_captchas(processed_captchas_dir, labels_csv)

    # Step 3: Load data and train the model
    X, y = load_data(processed_captchas_dir, labels_csv)
    model = train_model(X, y)

    # Save the model
    model.save("captcha_reader_model.h5")
    print("Model saved as captcha_reader_model.h5")
