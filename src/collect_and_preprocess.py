# collect_and_preprocess.py
import os
import cv2
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import numpy as np

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
        
        # Chuyển ảnh sang grayscale (đen trắng)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Lưu ảnh đã xử lý (chỉ chuyển thành ảnh đen trắng)
        file_path = os.path.join(save_dir, f'captcha_{i}.png')
        cv2.imwrite(file_path, gray)
        print(f"Processed and saved CAPTCHA {i} to {file_path}")
        
        driver.refresh()
        

    driver.quit()

# Main execution
if __name__ == "__main__":
    captcha_url = "https://id.ou.edu.vn/captcha"
    processed_captchas_dir = "D:/OU/Captcha/data"

    # Step 1: Collect and preprocess CAPTCHA images directly from the web
    collect_and_preprocess_captchas(captcha_url, num_captchas= 1000, save_dir=processed_captchas_dir)