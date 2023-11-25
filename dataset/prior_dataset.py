import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from PIL import Image

class PriorDatasetBase(Dataset):
    def __init__(self, image_text_data: pd.DataFrame):
        super().__init__()
        self.image_text_data = image_text_data

        # Define image transformation pipeline
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_text_data)

    def __getitem__(self, idx):
        # Retrieve image path and text for the given index
        image_path = self.image_text_data.iloc[idx]["image_path"]
        image = Image.open(image_path).convert("RGB")

        # Load and preprocess the image
        image_tensor = self.image_transform(image)
        text = self.image_text_data.iloc[idx]["text"]

        return image_tensor, text
    
class PriorDatasetModule(LightningDataModule):
    def __init__(self, text_image_data_path, batch_size = 8, num_workers = 0):
        super().__init__()

        # Store configuration parameters
        self.text_image_data_path = text_image_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Read and preprocess text-image data from CSV file
        self.text_image_data = pd.read_csv(self.text_image_data_path, usecols=["text", "label", "ViewPosition", "split", "image_path"])
        self.text_image_data = self.text_image_data.dropna(subset = ["text"])
        self.text_image_data = self.text_image_data.dropna(subset = ["image_path"])

        # Replace ".dcm" with ".jpg" in image paths
        self.text_image_data["image_path"] = self.text_image_data["image_path"].apply(lambda x: x.replace(".dcm", ".jpg"))

        # Split data into training and validation sets
        self.train_text_image_data = self.text_image_data[self.text_image_data["split"] == "train"]
        self.val_text_image_data = self.text_image_data[self.text_image_data["split"] == "validate"]

        # Create PriorDatasetBase instances for training and validation
        self.train_dataset = PriorDatasetBase(self.train_text_image_data)
        self.val_dataset = PriorDatasetBase(self.val_text_image_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    
    def dl_collate_fn(self, batch):
        return torch.stack([row[0] for row in batch]), list([row[1] for row in batch])
    