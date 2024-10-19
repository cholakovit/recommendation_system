import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class UserItemRatingDataset(Dataset):
    """
    This class is responsible for loading the dataset and providing data 
    in a format that PyTorch can use for training.
    """
    def __init__(self, file_path):
        df = pd.read_csv(file_path)

        # Convert user_id and item_id into categorical indices for better embeddings
        df['user_id'] = df['user_id'].astype('category').cat.codes
        df['item_id'] = df['item_id'].astype('category').cat.codes

        self.users = df['user_id'].values
        self.items = df['item_id'].values
        self.ratings = df['rating'].values

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.ratings)

    def __getitem__(self, idx):
        # Fetch the user, item, and rating for the given index
        return self.users[idx], self.items[idx], self.ratings[idx]

    @staticmethod
    def get_data_loaders(file_path, batch_size=64, train_split=0.8):
        """
        Creates DataLoaders for training and validation.
        """
        dataset = UserItemRatingDataset(file_path)
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Return DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        return train_loader, val_loader
