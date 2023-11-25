import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

# Define a base dataset class for language modeling
class LMDatasetBase(Dataset):
    def __init__(self, text_data: pd.DataFrame):
        super().__init__()
        self.text_data = text_data

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        # Retrieve text at the specified index
        text = self.text_data.iloc[idx]["text"]
        return text
    
# Define a LightningDataModule for handling the language modeling dataset
class LMDatasetModule(LightningDataModule):
    def __init__(self, data_path, batch_size = 8, num_workers = 0):
        super().__init__()

        # Store dataset-related information
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Read and preprocess the data from the provided CSV file
        self.data = pd.read_csv(self.data_path, usecols=["split", "text"])
        # Remove rows with text labeled as "none" and drop NaN values in the "text" column
        self.data = self.data.drop(self.data[self.data["text"] == "none"].index)
        self.data = self.data.dropna(subset = ["text"])

        # Split the data into training and validation sets
        self.train_data = self.data[self.data["split"] == "train"]
        self.val_data = self.data[self.data["split"] == "validate"]

        # Create instances of the LMDatasetBase for training and validation datasets
        self.train_dataset = LMDatasetBase(self.train_data)
        self.val_dataset = LMDatasetBase(self.val_data)

    # Define the method to get the training DataLoader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers = self.num_workers)
    
    # Define the method to get the validation DataLoader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers = self.num_workers)

