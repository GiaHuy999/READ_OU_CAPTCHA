
import os
import cv2
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import numpy as np

def collect_and_preprocess_captchas(url, num_captchas, save_dir):

    chrome_options = Options()
    chrome_options.add_argument("--headless")  

    service = Service('D:/OU/Captcha/chromedriver-win64/chromedriver.exe')  
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(url)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_captchas):
        captcha_element = driver.find_element(By.TAG_NAME, 'img')
        captcha_src = captcha_element.get_attribute('src')
        
        
        captcha_data = requests.get(captcha_src).content
        img = cv2.imdecode(np.frombuffer(captcha_data, np.uint8), cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        file_path = os.path.join(save_dir, f'captcha_{i}.png')
        cv2.imwrite(file_path, gray)
        print(f"Processed and saved CAPTCHA {i} to {file_path}")
        
        driver.refresh()
        

    driver.quit()

if __name__ == "__main__":
    captcha_url = "https://id.ou.edu.vn/captcha"
    processed_captchas_dir = "D:/OU/Captcha/data"
    collect_and_preprocess_captchas(captcha_url, num_captchas= 1000, save_dir=processed_captchas_dir)
