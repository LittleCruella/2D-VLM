import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset

import json
import torchvision.transforms as T # Renamed to T for brevity
import SimpleITK as sitk
from PIL import Image


class CLIPDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r', encoding='utf-8') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        # Get desired input size from args
        # args.input_image is a tuple (H, W) or (H, W, D) for 3D.
        # For 2D images, we only care about H and W.
        # Ensure args.input_image is passed from ModelArguments.
        # Example: input_image=(256, 256) or (256, 256, 128)
        # If it's (256, 256, 128) for 3D, and you are processing 2D,
        # you must decide the 2D size you want to resize to,
        # e.g., [args.input_image[0], args.input_image[1]]

        # Assuming args.input_image is a tuple (H, W) or (H, W, D)
        # We need 2D dimensions (H, W) for ViT2D
        if len(self.args.input_image) >= 2:
            target_image_size = (self.args.input_image[0], self.args.input_image[1])
        else:
            raise ValueError("input_image in ModelArguments must specify at least H and W dimensions (e.g., (256, 256) or (256, 256, 128)).")

        # Transforms for training
        train_transform = T.Compose(
            [
                # First, ensure image is a PIL Image or compatible for torchvision transforms
                # For medical images from SimpleITK, it's often a numpy array.
                # We'll convert to PIL Image during __getitem__ before these transforms.
                T.Resize(target_image_size), # Resize to fixed size
                T.RandomApply([T.RandomRotation(degrees=90)], p=0.5), # Equivalent to RandRotate90 for 2D
                T.RandomHorizontalFlip(p=0.10), # Equivalent to RandFlip spatial_axis=1
                T.RandomVerticalFlip(p=0.10), # Equivalent to RandFlip spatial_axis=0
                # For intensity transforms, torchvision doesn't have direct equivalents for RandScaleIntensity/RandShiftIntensity.
                # You might need custom transforms or libraries like Albumentations for these.
                # For simplicity here, we'll omit them or use basic color jitter if applicable (not for grayscale medical images).
                # If these are crucial, consider implementing them as custom torch.nn.Module or using Albumentations.
                # For now, let's add a placeholder for intensity adjustment if needed.
                # T.ColorJitter(brightness=0.1, contrast=0.1) # Not suitable for medical images often
                T.ToTensor(), # Convert to tensor (H, W) -> (C, H, W) where C is 1 for grayscale
                # Normalization is often applied after ToTensor. Add if needed.
                # T.Normalize(mean=[0.5], std=[0.5])
            ]
        )

        # Transforms for validation/test
        val_transform = T.Compose(
            [
                T.Resize(target_image_size), # Resize to fixed size
                T.ToTensor(), # Convert to tensor (H, W) -> (C, H, W)
                # T.Normalize(mean=[0.5], std=[0.5])
            ]
        )

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

                # Use SimpleITK to read the image
                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image) # np.ndarray (D, H, W) or (H, W)

                # Handle 3D image to 2D slice if necessary
                # if image.ndim == 3: # If image is 3D (D, H, W)
                #     # For training, randomly select a slice
                #     if self.mode == 'train':
                #         slice_idx = random.randint(0, image.shape[0] - 1)
                #     else: # For validation/test, pick the middle slice
                #         slice_idx = image.shape[0] // 2
                #     image = image[slice_idx, :, :] # Reduce to (H, W)

                # # Ensure image is 2D (H, W) for torchvision transforms.
                # # torchvision ToTensor expects (H, W) or (H, W, C) for numpy,
                # # and will convert to (C, H, W). For grayscale, it will be (1, H, W).
                # # Convert numpy array to PIL Image as torchvision transforms often operate on PIL Images.
                # # If your medical images have intensities outside 0-255, you might need to normalize
                # # them to 0-1 or 0-255 range before converting to PIL Image for best results.
                # # Or, apply transforms directly on numpy array and then convert to tensor manually.
                # # For simplicity, assuming image data is suitable for direct PIL conversion or normalization.

                # # Normalize image to [0, 255] and convert to uint8 for PIL, if it's float or out of range.
                # # This is a common step for medical images before applying standard image transforms.
                # if image.dtype != np.uint8:
                #     image_min = image.min()
                #     image_max = image.max()
                #     if image_max > image_min:
                #         image = ((image - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                #     else: # Handle cases where image is constant (e.g., all zeros)
                #         image = np.zeros_like(image, dtype=np.uint8)

                if image.ndim == 3:  # If image is 3D (D, H, W)
                    image = image.mean(-1)
                image_pil = Image.fromarray(image) # 'L' for grayscale

                # print(f"DEBUG: Loaded image from {image_abs_path} with shape {image.shape}, converted to PIL with size {image_pil.size}")
                # Apply transforms, including Resize
                image_tensor = self.transform(image_pil) # After transform, image_tensor will have shape (C, H, W) fixed

                # Ensure tensor has correct dimensions (e.g., C, H, W)
                # ViTEncoder expects (batch, C, H, W)

                raw_text = data["text"] # Get original text from JSON
                text = self.truncate_text(raw_text, self.args.max_length) # Use your function to process text

                text_tensor = self.tokenizer(
                    text,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length", # Very important for uniform padding
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0] # Get tensor (max_length,)
                attention_mask = text_tensor["attention_mask"][0] # Get tensor (max_length,)

                # Print statement to check shapes before returning
                # print(f"DEBUG: image_tensor.shape={image_tensor.shape}, input_id.shape={input_id.shape}, attention_mask.shape={attention_mask.shape}")

                ret = {
                    'image': image_tensor,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                # If an error occurs, try a different random index
                idx = random.randint(0, len(self.data_list) - 1)
        
        # If max_attempts are exhausted, raise an error
        raise ValueError(f"Failed to load a valid item after {max_attempts} attempts for index {idx}. Please check data integrity.")