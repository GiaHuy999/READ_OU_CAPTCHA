import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    img = cv2.resize(img, (100, 40))
    img = img.reshape((40, 100, 1)) / 255.0
    return np.expand_dims(img, axis=0)  

def decode_prediction(prediction, num_classes, num_characters_per_captcha):
    print("Prediction shape:", prediction.shape)  
    prediction = prediction.reshape((num_characters_per_captcha, num_classes))  
    decoded_label = []
    for i in range(num_characters_per_captcha):

        predicted_char_index = np.argmax(prediction[i])

        if predicted_char_index < 10:
            decoded_label.append(chr(predicted_char_index + ord('0'))) 
        elif predicted_char_index < 36:
            decoded_label.append(chr(predicted_char_index - 10 + ord('A')))  
        else:
            decoded_label.append(chr(predicted_char_index - 36 + ord('a'))) 
    return ''.join(decoded_label)

if __name__ == "__main__":
    
    model = load_model("captcha_reader_model.h5")
    num_classes = 62  
    num_characters_per_captcha = 4  

    image_path = "data\\captcha_0002.png"  
    X_new = process_image(image_path)
    prediction = model.predict(X_new)
    
    decoded_label = decode_prediction(prediction, num_classes, num_characters_per_captcha)
    print(f"Predicted CAPTCHA text: {decoded_label}")
