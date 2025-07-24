import csv
import random

def split_csv_into_two(input_csv_file, output_csv_file, val_ratio):
    """
    Tạo một file CSV mới chứa dữ liệu theo tỷ lệ từ file CSV gốc.

    Args:
        input_csv_file (str): Đường dẫn đến file CSV đầu vào.
        output_csv_file (str): Đường dẫn đến file CSV đầu ra.
        train_ratio (float): Tỷ lệ dữ liệu được giữ lại trong file đầu ra.
    """
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError("Tỷ lệ train phải nằm trong khoảng từ 0.0 đến 1.0.")

    try:
        with open(input_csv_file, 'r', encoding='utf-8') as infile:
            reader = list(csv.DictReader(infile))
            random.shuffle(reader)  # Xáo trộn dữ liệu để đảm bảo ngẫu nhiên

            total_items = len(reader)
            num_train = int(total_items * val_ratio)

            train_set = reader[:num_train]

            # Ghi tập dữ liệu vào file CSV đầu ra
            with open(output_csv_file, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=reader[0].keys())
                writer.writeheader()
                writer.writerows(train_set)

            print(f"Đã lưu tập dữ liệu với tỷ lệ {val_ratio} vào: {output_csv_file}")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Gọi hàm để tạo file CSV mới với tỷ lệ từ file gốc
split_csv_into_two(
    input_csv_file='data/vqa_rad/train_with_images.csv',
    output_csv_file='data/vqa_rad/val_with_images.csv',
    val_ratio=0.2
)