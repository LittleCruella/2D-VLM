import csv

def add_image_column_to_csv(input_csv_file, output_csv_file):
    """
    Thêm một cột 'image' vào file CSV với giá trị mỗi dòng là {số thứ tự mỗi dòng}.jpg.

    Args:
        input_csv_file (str): Đường dẫn đến file CSV đầu vào.
        output_csv_file (str): Đường dẫn đến file CSV đầu ra.
    """
    try:
        with open(input_csv_file, 'r', encoding='utf-8') as infile, open(output_csv_file, 'w', encoding='utf-8', newline='') as outfile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames + ['image']  # Thêm cột 'image'
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            writer.writeheader()
            for idx, row in enumerate(reader):
                row['image'] = f"vqa_rad/images/test/{idx}.jpg"  # Thêm giá trị cho cột 'image'
                writer.writerow(row)

        print(f"Đã thêm cột 'image' và lưu file mới tại: {output_csv_file}")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Gọi hàm để thêm cột 'image' vào file train.csv
add_image_column_to_csv(
    input_csv_file='data/vqa_rad/test.csv',
    output_csv_file='data/vqa_rad/test_with_images.csv'
)