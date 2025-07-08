import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset

import json
import monai.transforms as mtf
import SimpleITK as sitk
from monai.data import set_track_meta


class CLIPDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        # Lấy kích thước đầu vào mong muốn từ args
        # args.input_size là tuple (H, W, D) hoặc (H, W) cho 2D.
        # Với ảnh 2D, ta chỉ quan tâm đến H và W.
        # Đảm bảo args.input_size được truyền từ ModelArguments
        # Ví dụ: input_size=(256, 256) hoặc (256, 256, 128)
        # Nếu là (256, 256, 128) cho 3D, và bạn đang xử lý 2D,
        # thì bạn phải quyết định kích thước 2D mà bạn muốn resize về,
        # ví dụ [args.input_size[0], args.input_size[1]]
        
        # Giả sử args.input_size là một tuple (H, W) hoặc (H, W, D)
        # Chúng ta cần kích thước 2D là (H, W) cho ViT2D
        if len(self.args.input_image) >= 2:
            target_image_size = (self.args.input_image[0], self.args.input_image[1])
        else:
            raise ValueError("input_size in ModelArguments must specify at least H and W dimensions (e.g., (256, 256) or (256, 256, 128)).")

        # THÊM: Resize transform để đảm bảo tất cả ảnh có cùng kích thước
        # mtf.SpatialPad và mtf.Resize đều có thể dùng. Resize thường tốt hơn.
        # Ensure that Resize handles 2D images correctly.
        # channels_first=True means input is (C, H, W), which is what ToTensor produces.
        
        # Các transform cho huấn luyện
        train_transform = mtf.Compose(
            [
                # Đọc ảnh 3D, sau đó chọn 1 slice và squeeze về 2D,
                # sau đó mới áp dụng các transform 2D.
                # Hoặc nếu ảnh đã là 2D, chỉ cần đảm bảo kích thước.
                
                # THÊM: Resize ảnh về kích thước cố định trước khi apply các transform khác
                mtf.Resize(spatial_size=target_image_size),
                
                mtf.RandRotate90(prob=0.5, spatial_axes=(0, 1)), # Đảm bảo axes đúng cho (H, W)
                mtf.RandFlip(prob=0.10, spatial_axis=0), # Lật theo chiều H
                mtf.RandFlip(prob=0.10, spatial_axis=1), # Lật theo chiều W
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float), # Chuyển đổi thành tensor (H, W) -> (C, H, W)
                                                 # Monai ToTensor sẽ thêm chiều kênh nếu input không có (C, ...)
            ]
        )

        # Các transform cho validation/test
        val_transform = mtf.Compose(
            [
                # THÊM: Resize ảnh về kích thước cố định
                mtf.Resize(spatial_size=target_image_size),
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False) # Tắt theo dõi metadata của Monai

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
            self.data_list = self.data_list[:512]
        elif 'test' in mode:
            self.transform = val_transform
            self.data_list = self.data_list[:test_size]

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        # Giữ nguyên hàm này vì nó không liên quan đến lỗi ảnh
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image) # np.ndarray (D, H, W) hoặc (H, W)
                image = np.transpose(image, (2, 0, 1)) # (H, W, C) -> (C, H, W)
                print(f"DEBUG: Loaded image from {image_abs_path} with shape {image.shape}")
                # Xử lý ảnh 3D thành 2D nếu cần (nếu dữ liệu là 3D)
                if image.ndim == 3: # Nếu ảnh là 3D (D, H, W)
                    # Chọn một slice ở giữa D chiều hoặc ngẫu nhiên
                    # Ví dụ: chọn slice giữa
                    slice_idx = image.shape[0] // 2
                    image = image[slice_idx, :, :] # Giảm về (H, W)
                    # Hoặc chọn ngẫu nhiên nếu trong train mode
                    # if self.mode == 'train':
                    #     slice_idx = random.randint(0, image.shape[0] - 1)
                    #     image = image[slice_idx, :, :]
                    # else:
                    #     slice_idx = image.shape[0] // 2
                    #     image = image[slice_idx, :, :]

                # Đảm bảo ảnh có 1 kênh (C, H, W) hoặc (H, W) trước khi apply transform
                # Monai ToTensor sẽ thêm chiều kênh nếu input không có chiều đầu tiên là kênh.
                # Nếu ảnh của bạn là (H, W) sau khi đọc, ToTensor sẽ biến thành (1, H, W).
                # Nếu ảnh của bạn đã là (C, H, W), nó sẽ giữ nguyên.
                # Bạn không cần np.expand_dims(image, axis=0) ở đây nếu bạn muốn Monai tự thêm kênh.
                # Nếu ảnh gốc là (H, W), Monai ToTensor sẽ biến nó thành (1, H, W)
                # Nếu ảnh gốc là (C, H, W), Monai ToTensor sẽ giữ nguyên

                # Nếu ảnh của bạn là (H,W) và bạn muốn rõ ràng là (1, H, W) trước khi apply transforms:
                # if image.ndim == 2:
                #     image = np.expand_dims(image, axis=0) # Chuyển (H, W) thành (1, H, W)

                # Áp dụng các transform, bao gồm Resize
                image = self.transform(image) # Sau transform, image sẽ có shape (C, H, W) cố định

                # Đảm bảo tensor có đúng số chiều (vd: C, H, W)
                # ViTEncoder mong đợi (batch, C, H, W)

                # text = data["text"][:self.args.max_length] # Lấy trực tiếp từ JSON, bạn cần đảm bảo nó là chuỗi
                # Truncate text (nếu bạn sử dụng hàm `truncate_text` của mình, hãy bỏ comment)
                # text = self.truncate_text(data["text"], self.args.max_length)
                
                # Nếu bạn lấy text trực tiếp từ JSON, hãy đảm bảo nó là chuỗi và không quá dài
                raw_text = data["text"] # Lấy text gốc từ JSON
                text = self.truncate_text(raw_text, self.args.max_length) # Sử dụng hàm của bạn để xử lý text

                text_tensor = self.tokenizer(
                    text,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length", # Rất quan trọng để padding đồng đều
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0] # Lấy tensor (max_length,)
                attention_mask = text_tensor["attention_mask"][0] # Lấy tensor (max_length,)

                # Dòng print để kiểm tra shape trước khi trả về
                print(f"DEBUG: image.shape={image.shape}, input_id.shape={input_id.shape}, attention_mask.shape={attention_mask.shape}")

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)
                # Quan trọng: Nếu xảy ra lỗi, bạn phải return một giá trị hợp lệ để DataLoader không bị crash.
                # Cách tốt nhất là thử lại với một index ngẫu nhiên khác.
                # Tuy nhiên, nếu lỗi liên tục, vòng lặp vô hạn sẽ xảy ra.
                # Đảm bảo rằng bạn có một cơ chế thoát hoặc giới hạn số lần thử.
                # Với max_attempts, bạn nên raise lỗi nếu vượt quá giới hạn.
        
        # Nếu đã thử max_attempts lần mà vẫn lỗi, raise ValueError
        raise ValueError(f"Failed to load a valid item after {max_attempts} attempts for index {idx}. Please check data integrity.")