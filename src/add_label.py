import os
import csv

def label_captchas(labels_file):
    
    valid_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    output_dir = "D:\\OU\\Captcha\\READ_OU_CAPTCHA\\data"

    
    if not os.path.isdir(output_dir):
        print(f"The directory {output_dir} does not exist or is not a valid directory.")
        return

   
    png_files = [file for file in os.listdir(output_dir) if file.endswith('.png')]
    if not png_files:
        print("No .png files found in the specified directory.")
        return
    else:
        print(f"Found .png files: {png_files}")

    
    def sort_key(file_name):
        return int(file_name.split('_')[1].split('.png')[0])

    start_index = 0
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            if len(rows) > 1:
                last_row = rows[-1]
                last_file_name = last_row[0]
                start_index = int(last_file_name.split('_')[1].split('.png')[0])

    with open(labels_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if start_index == 0:
            writer.writerow(["file_name", "label"])

        for i, file_name in enumerate(sorted(png_files, key=sort_key)):
            if i >= start_index:
                
                label = input(f"Enter label for {file_name}: ")
                while not all(c in valid_characters for c in label):
                    print(f"Invalid label: {label}")
                    label = input("Enter the CAPTCHA label again: ")

                writer.writerow([file_name, label])

labels_file = 'labels.csv' 
label_captchas(labels_file)
