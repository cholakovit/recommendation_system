import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Load dataset (user-item interactions like user rating for movies)
df = pd.read_csv('user_ratings.csv')

# Convert user_id and item_id into categorical indicies
df['user_id'] = df['user_id'].astype('category').cat.codes
df['item_id'] = df['item_id'].astype('category').cat.codes

# Create training and validation datasets
train_size = int(0.8 * len(df))
train_data = df[:train_size]
val_data = df[train_size:]

# Custom Dataset Class
class UserItemRatingDataset(Dataset):
    def __init__(self, df):
        self.users = df['user_id'].values
        self.items = df['item_id'].values
        self.ratings = df['rating'].values

    def __len(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]
    
# Prepare DataLoaders
train_dataset = UserItemRatingDataset(train_data)
val_dataset = UserItemRatingDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Build the Recommendation Model Using Matrix Factorization
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=50):
        super(MatrixFactorization, self).__init__()

        # Embedding Layers for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)

        # Linear layer to predict the interaction score
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, user, item):
        # Extract embeddings
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)

        # Element-wise multiply user and item enbeddings (interaction features)
        interaction = user_embedded * item_embedded

        # Sum the product and predict the rating
        interaction = torch.sum(interaction, dim=1)

        return interaction
    
# Define model, loss, and optimizer
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()

model = MatrixFactorization(num_users, num_items, embedding_size=50)
optimizer = optim.adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for user, item, rating in train_loader:
        user = user.to(device)
        item = item.to(device)
        rating = rating.float().to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(user, item)
        loss = criterion(predictions, rating)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}')

# Validation (evaluate the model on the validation set)
model.eval()
with torch.no_grad():
    val_loss = 0
    for user, item, rating in val_loader:
        user = user.to(device)
        item = item.to(device)
        rating = rating.float().to(device)

        predictions = model(user, item)
        loss = criterion(predictions, rating)

        val_loss += loss.item()

    print(f'Validation Loss: {val_loss / len(val_loader):.4f}')

# Example: Recommend items for a specific user (user_id=0)
def recommend_items(user_id, model, num_recommendations=5):
    user_id = torch.tensor(0, num_items).to(device)

    # Get predictions for all tems for this user
    item_ids = torch.arange(0, num_items).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(user_id.expand_as(item_ids), item_ids)

    # Sort items by predicted rating
    top_items = torch.argsort(predictions, descending=True)[:num_recommendations]

    return top_items.cpu().numpy()

recommend_items = recommend_items(user_id=0, model=model, num_recommendations=5)
print(f'Recommended items for user 0: {recommend_items}')

# Save the model
torch.save(model.state_dict(), 'recommnedation_model.pth')

# Load the model for inference
model = MatrixFactorization(num_users, num_items)
model.load_state_dict(torch.load('recommnedation_model.pth'))
modal.eval()