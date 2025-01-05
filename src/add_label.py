import os
import csv

def label_captchas(labels_file):
    # Xác định các ký tự hợp lệ
    valid_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

    # Nhập thư mục chứa ảnh CAPTCHA
    output_dir = "D:\\OU\\Captcha\\data"  # Chú ý dấu \ cần escape (\\) hoặc dùng dấu / thay vì \

    # Kiểm tra thư mục đầu ra có hợp lệ không
    if not os.path.isdir(output_dir):
        print(f"The directory {output_dir} does not exist or is not a valid directory.")
        return

    # Kiểm tra các file .png trong thư mục
    png_files = [file for file in os.listdir(output_dir) if file.endswith('.png')]
    if not png_files:
        print("No .png files found in the specified directory.")
        return
    else:
        print(f"Found .png files: {png_files}")

    # Hàm tùy chỉnh để sắp xếp theo số trong tên file
    def sort_key(file_name):
        # Lấy số trong tên file, ví dụ: 'captcha_1.png' -> 1
        return int(file_name.split('_')[1].split('.png')[0])

    # Kiểm tra nếu file CSV đã tồn tại, nếu có thì lấy số thứ tự ảnh tiếp theo
    start_index = 0
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            if len(rows) > 1:
                # Lấy chỉ số của ảnh cuối cùng từ file CSV
                last_row = rows[-1]
                last_file_name = last_row[0]
                # Giả sử tên file có dạng 'captcha_1.png', 'captcha_2.png', etc.
                start_index = int(last_file_name.split('_')[1].split('.png')[0]) + 1

    # Mở file CSV để ghi nhãn
    with open(labels_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if start_index == 0:
            writer.writerow(["file_name", "label"])

        # Lặp qua tất cả các file trong thư mục đầu ra, bắt đầu từ start_index
        for i, file_name in enumerate(sorted(png_files, key=sort_key)):
            if i >= start_index:
                print(f"Displaying: {file_name}")
                
                # Yêu cầu nhập nhãn
                label = input(f"Enter the CAPTCHA label for {file_name}: ")

                # Kiểm tra nếu nhãn chứa chỉ các ký tự hợp lệ
                while not all(c in valid_characters for c in label):
                    print(f"Invalid label: {label}. Only characters from {valid_characters} are allowed.")
                    label = input("Enter the CAPTCHA label again: ")

                # Ghi nhãn hợp lệ vào file CSV
                writer.writerow([file_name, label])
                print(f"Label for {file_name}: {label}")

# Ví dụ gọi hàm
labels_file = 'labels.csv'  # Đường dẫn đến file CSV lưu nhãn
label_captchas(labels_file)
