import os
import json
import logging
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from PIL import Image
# from datasets import Dataset, Features, Value, Image as HFImage
from datasets import Dataset, Features, Image as HFImage, Value
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
import logging
import re
import json
from difflib import SequenceMatcher

class InvoiceDatasetCreator:
    def __init__(self, images_folder: str, labels_folder: str, output_folder: str = "output"):
        self.images_path = Path(images_folder)
        self.labels_path = Path(labels_folder)
        self.output_path = Path(output_folder)
        self.output_path.mkdir(exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def clean_text(self, text: str) -> str:
        """Clean text by removing extra spaces and normalizing punctuation."""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(' .', '.')
        text = text.replace('..', '.')
        return text.strip()

    def are_strings_similar(self, str1: str, str2: str, threshold: float = 0.9) -> bool:
        """Check if two strings are similar using sequence matcher."""
        str1 = self.clean_text(str1)
        str2 = self.clean_text(str2)
        
        if str1 == str2:
            return True
        
        similarity = SequenceMatcher(None, str1, str2).ratio()
        return similarity >= threshold

    def process_json_labels(self, label_json: Union[str, dict]) -> dict:
        """Process JSON labels to remove duplicates and clean format."""
        if isinstance(label_json, str):
            data = json.loads(label_json)
        else:
            data = label_json
        
        cleaned_data = {}
        
        for key, value in data.items():
            # Extract all values without bounding boxes
            values = [self.clean_text(item[0]) for item in value]
            
            # Remove duplicates and similar values
            unique_values = []
            for val in values:
                if not any(self.are_strings_similar(val, existing_val) for existing_val in unique_values):
                    unique_values.append(val)
            
            # Assign the value(s)
            if len(unique_values) == 1:
                cleaned_data[key] = unique_values[0]
            else:
                cleaned_data[key] = unique_values
        
        return cleaned_data
        
    def validate_invoice_file_pair(self, image_file: Path) -> Tuple[bool, str, str]:
        """Validate if an invoice image file has a corresponding label file."""
        try:
            pattern = r'^\d+_Invoice_page_\d+\.png$'
            print("image_file.name", image_file.stem)
            if not re.match(pattern, image_file.name):
                logging.warning(f"Invalid file name format: {image_file.name}")
                return False, '', ''
                
            label_file = self.labels_path / f"{image_file.stem}_labels.txt"
            
            if not label_file.exists():
                logging.warning(f"Missing label file: {label_file}")
                return False, '', ''
                
            with label_file.open('r', encoding='utf-8') as f:
                label_content = f.read().strip()
                
            if not label_content:
                logging.warning(f"Empty label file: {label_file}")
                return False, '', ''
            
            # Process the JSON content
            try:
                processed_label = self.process_json_labels(label_content)
                processed_label_str = json.dumps(processed_label)
                return True, str(image_file), processed_label_str
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON in label file {label_file}: {str(e)}")
                return False, '', ''
                
        except Exception as e:
            logging.error(f"Error processing {image_file}: {str(e)}")
            return False, '', ''
            
    def process_invoice_files(self) -> Tuple[List[str], List[str]]:
        """Process all invoice image and label files."""
        image_paths = []
        labels = []
        
        image_files = list(self.images_path.glob('*_Invoice_page_*.png'))
        
        if not image_files:
            logging.warning("No invoice images found in the specified directory")
            return [], []
            
        for image_file in image_files:
            result = self.validate_invoice_file_pair(image_file)
            if result[0]:
                image_paths.append(result[1])
                labels.append(result[2])
        
        logging.info(f"Found {len(image_paths)} valid invoice-label pairs")
        print("length of img and path", len(image_paths), len(labels))
        return image_paths, labels
      
    def load_image(self, image_path: str):
        """Load and convert image to RGB format."""
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None

    def format_data_entry(self, image_path: str, label_data: dict) -> dict:
        """Format data entry in the required structure."""
        image = self.load_image(image_path)
        if not image:
            return None
        
        # with open(label_path, 'r', encoding='utf-8') as f:
        #     label_data = json.load(f)
        
        return {
            "image": image,
            "image_id": Path(image_path).stem,
            "label": json.dumps(label_data)
        }

    def create_dataset(self, chunk_size=50):
        """Create and save the dataset in Hugging Face's Dataset format with chunked GPU processing."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = transforms.Compose([
            transforms.ToTensor(),           # Converts PIL image to Tensor
            transforms.Lambda(lambda x: x.to(device))  # Move tensor directly to GPU
        ])

        dataset_entries = []

        image_files, label_files = self.process_invoice_files()

        if not image_files or not label_files:
            logging.error("No image or label files found. Dataset creation aborted.")
            return

        saved_paths = []
        c = 0

        for i in range(0, len(image_files), chunk_size):
            batch_image_files = image_files[i:i + chunk_size]
            batch_label_files = label_files[i:i + chunk_size]

            batch_entries = []
            for image_file, label_file in zip(batch_image_files, batch_label_files):
                entry = self.format_data_entry(str(image_file), str(label_file))
                if entry:
                    try:
                        tensor_image = transform(entry["image"])
                        entry["image"] = transforms.ToPILImage()(tensor_image.cpu())  # Back to PIL
                    
                        batch_entries.append(entry)
                    except Exception as e:
                        logging.error(f"Error processing image {image_file}: {str(e)}")

            # Skip saving if the current batch is empty
            if not batch_entries:
                continue
            else:
                print("length of the batch_entries is:", len(batch_entries))
                print("batch 1 is", batch_entries[0])

            # Convert tensors back to list form for Hugging Face Dataset
            dataset = Dataset.from_dict({
                "image": [entry["image"] for entry in batch_entries],
                "image_id": [entry["image_id"] for entry in batch_entries],
                "label": [entry["label"] for entry in batch_entries],
            }, features=Features({
                "image": HFImage(),
                "image_id": Value("string"),
                "label": Value("string")
            }))

            c += 1
            dataset_path = str(self.output_path / f"invoice_dataset_{c}")
            dataset.save_to_disk(dataset_path)
            saved_paths.append(dataset_path)
            logging.info(f"Dataset chunk {c} successfully saved to {dataset_path}")

            # Clear GPU memory after processing each chunk
            torch.cuda.empty_cache()

        if not saved_paths:
            logging.error("No valid invoice-label pairs found. Dataset creation aborted.")
            return None

        return saved_paths


def main():
    creator = InvoiceDatasetCreator(
        images_folder="/home/data_science/mani/table_extraction/llama-vision-finetuning/data/complete_invoice_data/images",
        labels_folder="/home/data_science/mani/table_extraction/llama-vision-finetuning/data/complete_invoice_data/Master_Labels",
        output_folder="/home/data_science/mani/table_extraction/llama-vision-finetuning/vision_kd/vision_data"
    )
    creator.create_dataset()

if __name__ == "__main__":
    main()