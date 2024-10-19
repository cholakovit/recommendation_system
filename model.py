import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    """
    This class defines the collaborative filtering model using matrix factorization.
    We create user and item embeddings and predict interaction scores between them.
    """
    def __init__(self, num_users, num_items, embedding_size=50):
        super(MatrixFactorization, self).__init__()
        
        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        # Fully connected layer for final prediction
        self.fc = nn.Linear(embedding_size, 1)
    
    def forward(self, user, item):
        # Get user and item embeddings
        user_embedded = self.user_embedding(user)
        item_embedded = self.item_embedding(item)
        
        # Calculate interaction score by element-wise multiplication
        interaction = user_embedded * item_embedded
        
        # Sum the interactions along the embedding dimensions
        interaction = torch.sum(interaction, dim=1)
        
        return interaction
