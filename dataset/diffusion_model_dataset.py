import torch
import ast
import pandas as pd
from transformers import CLIPVisionModel
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from PIL import Image

# Dataset class for the diffusion model
class DiffusionDatasetBase(Dataset):
    def __init__(self, image_text_data: pd.DataFrame, clip_reference_data: pd.DataFrame):
        super().__init__()
        self.image_text_data = image_text_data
        self.clip_reference_data = clip_reference_data

        # Load CLIPVisionModel for processing clip images
        self.clipvisionmodel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clipvisionmodel.requires_grad_(False)


    # Transform input image to tensor
    def image_transform(self, image, size):
        image_transform = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        processed_image = image_transform(image)
        return processed_image

    def __len__(self):
        return len(self.image_text_data)

    def __getitem__(self, idx):
        # Load and preprocess input image
        image_path = self.image_text_data.iloc[idx]["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image, 256)

        # Extract and process label
        label = ast.literal_eval(self.image_text_data.iloc[idx]["label"])
        label = torch.tensor(label)

        # Check if clip image exists for the given label
        clip_image_exist = False
        for i in range(self.clip_reference_data.shape[0]):
            row_label = self.clip_reference_data.iloc[i]["label"]
            if row_label == self.image_text_data.iloc[idx]["label"]:
                clip_image_exist = True
                clip_image_path = self.clip_reference_data.iloc[i]["image_path"]
                break

        # Load and preprocess clip image if it exists, otherwise create a tensor of zeros
        if clip_image_exist:
            clip_image = Image.open(clip_image_path).convert("RGB")
            clip_image_tensor = self.image_transform(clip_image, 224)
            clip_image_tensor = self.clipvisionmodel(clip_image_tensor.unsqueeze(0))[1]
        else:
            clip_image_tensor = torch.zeros(1,768)

        return image_tensor, label, clip_image_tensor
    
# LightningDataModule for managing the diffusion dataset
class DiffusionDatasetModule(LightningDataModule):
    def __init__(self, text_image_data_path, batch_size = 8, num_workers = 0):
        super().__init__()

        self.text_image_data_path = text_image_data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Read and preprocess the text image data
        self.text_image_data = pd.read_csv(self.text_image_data_path, usecols=["label", "ViewPosition", "split", "image_path"])
        self.text_image_data = self.text_image_data.dropna(subset = ["label"])
        self.text_image_data = self.text_image_data.dropna(subset = ["image_path"])

        # Filter data for "PA" view position
        self.text_image_data = self.text_image_data[self.text_image_data["ViewPosition"] == "PA"]

        # Split the data into training and validation sets
        self.train_data = self.text_image_data.loc[self.text_image_data["image_path"].str.contains("/p10/|/p11/|/p12/|/p13/|/p14/|/p15/|/p16/|/p17/|/p18/",na=False)]
        self.train_data_info_text = self.train_data[self.train_data["split"] == "train"]
        
        self.val_data_info_text = self.train_data[self.train_data["split"] == "validate"]   

        # Create instances of the DiffusionDatasetBase for training and validation
        self.train_dataset = DiffusionDatasetBase(self.train_data_info_text, self.train_data_info_text)
        self.val_dataset = DiffusionDatasetBase(self.val_data_info_text, self.train_data_info_text)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    
    def dl_collate_fn(self, batch):
        return torch.stack([row[0] for row in batch]), torch.stack([row[1] for row in batch]), torch.stack([row[2] for row in batch])

